# app.py
import os, io, base64, time, yaml, requests, uuid
from PIL import Image
import gradio as gr
from requests.exceptions import ConnectionError, Timeout, HTTPError
from typing import Any, Dict

# NEW: import and init limiter
from ratelimits import RateLimiter  # ensure ratelimiter.py is in the Space root
RL = RateLimiter()

# frontend-only: call your backend (RunPod/pod/etc.)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7861").rstrip("/")
TOKEN = os.getenv("RUNPOD_API_TOKEN", "xxxxx")

headers = {"Authorization": f"Bearer {TOKEN}"}
print(f"[HF] BACKEND_URL resolved to: {BACKEND_URL}")

# sample prompts
SAMPLE_PROMPTS = [
    "a high-resolution spiral galaxy with blue star-forming arms and a bright yellow core",
    "a crimson emission nebula with dark dust lanes and scattered newborn stars",
    "a ringed gas giant with visible storm bands and subtle shadow on rings",
    "an accretion disk around a black hole with relativistic jets, high contrast",
]

# default UI values if no YAML
cfg = {
    "height": 512,
    "width": 512,
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": 1234,
    "eta": 0,
}

# ---- health check ----
def check_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", headers=headers, timeout=40)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "ok":
            return "backend=READY"
    except Exception:
        pass
    return "backend=DOWN"

def b64_to_img(s: str):
    data = base64.b64decode(s)
    return Image.open(io.BytesIO(data)).convert("RGB")

# NEW: helper to get a stable client key
def _client_ip(req: gr.Request) -> str:
    # prefer real IP if present; fallback to client.host or session hash
    return (
        getattr(req.state, "real_ip", None)
        or (req.client.host if req.client else None)
        or getattr(req, "session_hash", None)
        or "unknown"
    )

# NEW: ensure session dict shape expected by RateLimiter
def _ensure_session(state: Dict[str, Any]) -> Dict[str, Any]:
    if not state or not isinstance(state, dict):
        state = {}
    state.setdefault("session_id", str(uuid.uuid4()))
    state.setdefault("count", 0)
    state.setdefault("started_at", time.time())
    return state

def _blank(w: int, h: int, rgb=(30, 30, 30)):
    return Image.new("RGB", (w, h), rgb)

def _infer(p, st, sc, h, w, sd, et, session_state: Dict[str, Any], request: gr.Request):
    # normalize numeric inputs
    h = int(h)
    w = int(w)

    # NEW: front-gate
    session_state = _ensure_session(session_state)
    ip = _client_ip(request)
    ok, reason = RL.pre_check(ip, session_state)
    if not ok:
        return _blank(w, h), _blank(w, h), f"[rate-limit] {reason}", session_state

    start_t = time.time()
    payload = {
        "prompt": p,
        "steps": int(st),
        "scale": float(sc),
        "height": h,
        "width": w,
        "seed": str(sd),
        "eta": float(et),
        # propagate session id downstream if backend also tracks
        "session_id": session_state.get("session_id"),
    }

    try:
        r = requests.post(f"{BACKEND_URL}/infer", headers=headers, json=payload, timeout=300)

        # backend throttle passthrough
        if r.status_code == 429:
            RL.post_consume(ip, max(0.0, time.time() - start_t))  # still account a tiny cost
            out = r.json()
            new_sid = out.get("session_id", session_state.get("session_id"))
            session_state["session_id"] = new_sid
            msg = out.get("error", "rate limited by backend")
            return _blank(w, h), _blank(w, h), msg, session_state

        r.raise_for_status()
        out = r.json()

        base_img = b64_to_img(out["base_image"])
        lora_img = b64_to_img(out["lora_image"])

        # sync session id if backend rotates it
        new_sid = out.get("session_id", session_state.get("session_id"))
        session_state["session_id"] = new_sid

        # NEW: only on success, increment session count
        session_state["count"] = int(session_state.get("count", 0)) + 1

        RL.post_consume(ip, max(0.0, time.time() - start_t))
        return base_img, lora_img, out.get("status", "ok"), session_state

    except ConnectionError:
        RL.post_consume(ip, max(0.0, time.time() - start_t))
        return _blank(w, h, (120, 50, 50)), _blank(w, h, (120, 50, 50)), \
               "Backend not reachable (connection refused). Start the backend and retry.", session_state

    except Timeout:
        RL.post_consume(ip, max(0.0, time.time() - start_t))
        return _blank(w, h, (120, 50, 50)), _blank(w, h, (120, 50, 50)), \
               "Backend took too long. Please try again later.", session_state

    except HTTPError as e:
        RL.post_consume(ip, max(0.0, time.time() - start_t))
        return _blank(w, h, (120, 50, 50)), _blank(w, h, (120, 50, 50)), \
               f"Backend returned HTTP Error: {e.response.status_code}", session_state

    except Exception as e:
        RL.post_consume(ip, max(0.0, time.time() - start_t))
        return _blank(w, h, (120, 50, 50)), _blank(w, h, (120, 50, 50)), \
               f"Unknown client error: {e}", session_state

def build_ui():
    with gr.Blocks(title="Astro-Diffusion: Base vs LoRA") as demo:
        session_state = gr.State(value={})  # NEW: dict state

        status_lbl = gr.Markdown("checking backend...")

        gr.HTML(
            """
            <style>
            .astro-header {
                background: linear-gradient(90deg, #0f172a 0%, #1d4ed8 50%, #0ea5e9 100%);
                padding: 0.9rem 1rem 0.85rem 1rem;
                border-radius: 0.6rem;
                margin-bottom: 0.9rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
            }
            .astro-title {
                color: #ffffff !important;
                margin: 0;
                font-weight: 700;
                letter-spacing: 0.01em;
                font-size: 1.4rem;
            }
            .astro-sub {
                color: #ffffff !important;
                margin: 0.3rem 0 0 0;
                font-style: italic;
                font-size: 0.9rem;
            }
            .astro-note {
                color: #ffffff !important;
                margin: 0.25rem 0 0.25rem 0;
                font-size: 0.8rem;
                opacity: 0.9;
            }
            .astro-link {
                margin-top: 0.55rem;
            }
            .astro-link a {
                color: #ffffff !important;
                text-decoration: underline;
                font-size: 0.78rem;
            }
            .astro-badge {
                background: #facc15;
                color: #0f172a;
                padding: 0.4rem 1.05rem;
                border-radius: 9999px;
                font-weight: 700;
                white-space: nowrap;
                font-size: 0.95rem;
            }
            .prompt-panel {
                background: #e8fff4;
                padding: 0.5rem 0.5rem 0.2rem 0.5rem;
                border-radius: 0.5rem;
                margin-bottom: 0.5rem;
            }
            .gradio-container label,
            label,
            .gradio-container [class*="label"],
            .gradio-container [class^="svelte-"][class*="label"],
            .gradio-container .block p > label {
                color: #000000 !important;
                font-weight: 600;
            }
            .gradio-container [data-testid="block-label"],
            .gradio-container [data-testid="block-label"] * {
                color: #000000 !important;
                font-weight: 600;
            }
            </style>
            <div class="astro-header">
                <div>
                    <h2 class="astro-title">Astro-Diffusion : Base SD vs custom LoRA</h2>
                    <p class="astro-sub">Video generation and more features coming up..!</p>
                    <p class="astro-note">Shared hourly/daily limits globally for this demo. Please use sparingly.</p>
                    <p class="astro-note">
                        The backend involves cold-start (serverless), so please refresh the browser until the backend status (shown top left) is READY.
                    </p>
                    <p class="astro-note">
                        Some errors may not be related to the model itself and may originate from the hosting platform.
                    </p>
                    <p class="astro-note">
                        I can only support the budget for a limited number of demos. Please check out the recorded
                        <a href="https://www.youtube.com/watch?v=mphpXnJV6yo" target="_blank" rel="noreferrer noopener">
                            demo
                        </a>.
                    </p>
                    <p class="astro-link">
                        <a href="https://github.com/KSV2001/astro_diffusion" target="_blank" rel="noreferrer noopener">
                            Visit Srivatsava's GitHub repo
                        </a>
                    </p>
                </div>
                <div class="astro-badge">by Srivatsava Kasibhatla</div>
            </div>
            """
        )

        with gr.Group(elem_classes=["prompt-panel"]):
            sample_dropdown = gr.Dropdown(
                choices=SAMPLE_PROMPTS,
                value=SAMPLE_PROMPTS[0],
                label="Sample prompts",
            )
            prompt = gr.Textbox(
                value=SAMPLE_PROMPTS[0],
                label="Prompt",
            )

        sample_dropdown.change(fn=lambda x: x, inputs=sample_dropdown, outputs=prompt)

        with gr.Row():
            steps = gr.Slider(10, 60, value=cfg.get("num_inference_steps", 30), step=1, label="Steps")
            scale = gr.Slider(1.0, 12.0, value=cfg.get("guidance_scale", 7.5), step=0.5, label="Guidance")
            height = gr.Number(value=min(int(cfg.get("height", 512)), 512), label="Height", minimum=32, maximum=512)
            width = gr.Number(value=min(int(cfg.get("width", 512)), 512), label="Width", minimum=32, maximum=512)
            seed = gr.Textbox(value=str(cfg.get("seed", 1234)), label="Seed")
            eta = gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Eta")

        btn = gr.Button("Generate")
        out_base = gr.Image(label="Base Model Output")
        out_lora = gr.Image(label="LoRA Model Output")
        status = gr.Textbox(label="Status", interactive=False)

        # pass session_state; Gradio injects request automatically
        btn.click(
            _infer,
            [prompt, steps, scale, height, width, seed, eta, session_state],
            [out_base, out_lora, status, session_state],
        )

        demo.load(fn=check_backend, inputs=None, outputs=status_lbl)

    return demo

if __name__ == "__main__":
    interface = build_ui()
    port = int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "8080")))
    share = os.getenv("GRADIO_PUBLIC_SHARE", "True").lower() == "true"
    interface.launch(server_name="0.0.0.0", server_port=port, share=share)
