# ratelimits.py
import time
import threading
import os
from typing import Tuple, Optional, Dict, Any
import logging

log = logging.getLogger("ratelimits")
if not log.handlers:
    logging.basicConfig(
        level=os.getenv("AD_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


class RateLimiter:
    """
    In-memory, multi-scope limiter.

    Scopes:
      - per-session
      - per-IP: hour, day
      - global: hour, day, month
      - time/cost: per-IP day, global day/month

    All counters reset on container restart.
    """

    def __init__(self):
        # ---------------- session limits ----------------
        self.per_session_max_req = int(os.getenv("AD_SESSION_MAX_REQ", "5"))
        self.per_session_max_age = int(os.getenv("AD_SESSION_MAX_AGE_SEC", str(15 * 60)))  # 15 min

        # ---------------- per-IP limits ----------------
        # relaxed defaults so you don't trip these while testing
        self.per_ip_max_req_hour = int(os.getenv("AD_IP_MAX_REQ_HOUR", "200"))
        self.per_ip_max_req_day = int(os.getenv("AD_IP_MAX_REQ_DAY", "1000"))
        self.per_ip_max_active_sec_day = int(
            os.getenv("AD_IP_MAX_ACTIVE_SEC_DAY", str(60 * 60))
        )  # 1h active time per day

        # ---------------- global limits ----------------
        self.global_max_req_hour = int(os.getenv("AD_GLOBAL_MAX_REQ_HOUR", "50"))
        self.global_max_req_day = int(os.getenv("AD_GLOBAL_MAX_REQ_DAY", "100"))
        self.global_max_req_month = int(os.getenv("AD_GLOBAL_MAX_REQ_MONTH", "500"))
        self.global_max_active_sec_day = int(
            os.getenv("AD_GLOBAL_MAX_ACTIVE_SEC_DAY", str(6 * 60 * 60))
        )  # 6h

        # ---------------- cost limits ----------------
        self.cost_per_sec = float(os.getenv("AD_COST_PER_SEC", "0.0005"))
        self.daily_cost_limit = float(os.getenv("AD_DAILY_COST_LIMIT", "5.0"))
        self.monthly_cost_limit = float(os.getenv("AD_MONTHLY_COST_LIMIT", "10.0"))

        self._lock = threading.Lock()

        # per-ip buckets
        self._ip_hour: Dict[str, Dict[str, float]] = {}
        self._ip_day: Dict[str, Dict[str, float]] = {}

        # global buckets
        now = time.time()
        self._global_hour = {"count": 0, "reset_at": now + 3600}
        self._global_day = {
            "count": 0,
            "active_sec": 0.0,
            "cost": 0.0,
            "reset_at": now + 86400,
        }
        self._global_month_req = {"count": 0, "reset_at": now + 30 * 86400}
        self._global_month_cost = {"cost": 0.0, "reset_at": now + 30 * 86400}

    # -------------------------------------------------
    # helpers
    # -------------------------------------------------
    @staticmethod
    def _fmt_wait(seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        if seconds >= 3600:
            h = seconds / 3600.0
            return f"{h:.1f} hours"
        elif seconds >= 60:
            m = seconds / 60.0
            return f"{m:.1f} minutes"
        else:
            return f"{seconds:.0f} seconds"

    def _get_ip_hour_bucket(self, ip: str):
        now = time.time()
        b = self._ip_hour.get(ip)
        if b is None or now >= b["reset_at"]:
            b = {"count": 0, "reset_at": now + 3600}
            self._ip_hour[ip] = b
        return b

    def _get_ip_day_bucket(self, ip: str):
        now = time.time()
        b = self._ip_day.get(ip)
        if b is None or now >= b["reset_at"]:
            b = {"count": 0, "active_sec": 0.0, "reset_at": now + 86400}
            self._ip_day[ip] = b
        return b

    def _get_global_hour(self):
        now = time.time()
        g = self._global_hour
        if now >= g["reset_at"]:
            g["count"] = 0
            g["reset_at"] = now + 3600
        return g

    def _get_global_day(self):
        now = time.time()
        g = self._global_day
        if now >= g["reset_at"]:
            g["count"] = 0
            g["active_sec"] = 0.0
            g["cost"] = 0.0
            g["reset_at"] = now + 86400
        return g

    def _get_global_month_req(self):
        now = time.time()
        g = self._global_month_req
        if now >= g["reset_at"]:
            g["count"] = 0
            g["reset_at"] = now + 30 * 86400
        return g

    def _get_global_month_cost(self):
        now = time.time()
        g = self._global_month_cost
        if now >= g["reset_at"]:
            g["cost"] = 0.0
            g["reset_at"] = now + 30 * 86400
        return g

    # -------------------------------------------------
    # public API
    # -------------------------------------------------
    def pre_check(
        self,
        ip: str,
        session_state: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Check count/time/cost limits BEFORE doing work.
        session_state is expected to have:
            {
              "count": int,
              "started_at": float
            }
        """
        now = time.time()
        with self._lock:
            # ---- session checks ----
            sess_count = int(session_state.get("count", 0))
            sess_started = float(session_state.get("started_at", now))
            session_age = now - sess_started

            # too old
            if session_age > self.per_session_max_age:
                msg = "session time cap reached. refresh the tab to start a new session."
                log.warning(f"[rl][session] expired ip={ip} age={session_age:.1f}s")
                return False, msg

            # too many requests in this session
            if sess_count >= self.per_session_max_req:
                msg = f"session request cap reached ({self.per_session_max_req}). refresh the tab to start a new session."
                log.warning(f"[rl][session] cap ip={ip} sess_count={sess_count}")
                return False, msg

            # ---- per-IP checks ----
            ip_h = self._get_ip_hour_bucket(ip)
            if ip_h["count"] >= self.per_ip_max_req_hour:
                remaining = ip_h["reset_at"] - now
                wait_str = self._fmt_wait(remaining)
                return (
                    False,
                    f"ip hourly cap {self.per_ip_max_req_hour} reached. try again in {wait_str}",
                )

            ip_d = self._get_ip_day_bucket(ip)
            if ip_d["count"] >= self.per_ip_max_req_day:
                remaining = ip_d["reset_at"] - now
                wait_str = self._fmt_wait(remaining)
                return (
                    False,
                    f"ip daily cap {self.per_ip_max_req_day} reached. try again in {wait_str}",
                )
            if ip_d["active_sec"] >= self.per_ip_max_active_sec_day:
                remaining = ip_d["reset_at"] - now
                wait_str = self._fmt_wait(remaining)
                return (
                    False,
                    f"ip daily active time cap {self.per_ip_max_active_sec_day} sec reached. try again in {wait_str}",
                )

            # ---- global checks ----
            g_h = self._get_global_hour()
            if g_h["count"] >= self.global_max_req_hour:
                remaining = g_h["reset_at"] - now
                wait_str = self._fmt_wait(remaining)
                return (
                    False,
                    f"global hourly cap {self.global_max_req_hour} reached. try again in {wait_str}",
                )

            g_d = self._get_global_day()
            if g_d["count"] >= self.global_max_req_day:
                remaining = g_d["reset_at"] - now
                wait_str = self._fmt_wait(remaining)
                return (
                    False,
                    f"global daily cap {self.global_max_req_day} reached. try again in {wait_str}",
                )
            if g_d["active_sec"] >= self.global_max_active_sec_day:
                remaining = g_d["reset_at"] - now
                wait_str = self._fmt_wait(remaining)
                return (
                    False,
                    f"global daily active time cap {self.global_max_active_sec_day} sec reached. try again in {wait_str}",
                )
            if g_d["cost"] >= self.daily_cost_limit:
                remaining = g_d["reset_at"] - now
                wait_str = self._fmt_wait(remaining)
                return (
                    False,
                    f"global daily cost cap {self.daily_cost_limit} reached. try again in {wait_str}",
                )

            g_m_req = self._get_global_month_req()
            if g_m_req["count"] >= self.global_max_req_month:
                remaining = g_m_req["reset_at"] - now
                wait_str = self._fmt_wait(remaining)
                return (
                    False,
                    f"global monthly cap {self.global_max_req_month} reached. try again in {wait_str}",
                )

            g_m_cost = self._get_global_month_cost()
            if g_m_cost["cost"] >= self.monthly_cost_limit:
                remaining = g_m_cost["reset_at"] - now
                wait_str = self._fmt_wait(remaining)
                return (
                    False,
                    f"global monthly cost cap {self.monthly_cost_limit} reached. try again in {wait_str}",
                )

            # ---- all clear -> increment request-based buckets ----
            ip_h["count"] += 1
            ip_d["count"] += 1

            g_h["count"] += 1
            g_d["count"] += 1
            g_m_req["count"] += 1

            # IMPORTANT:
            # do NOT increment session here;
            # api.py does it after a successful generation
            session_state.setdefault("started_at", now)

            log.info(f"[rl][session] ok ip={ip} sess_count={sess_count}")
            return True, None

    def post_consume(
        self,
        ip: str,
        duration_sec: float,
    ) -> None:
        """
        Update time-based and cost-based buckets AFTER doing work.
        """
        cost = duration_sec * self.cost_per_sec
        with self._lock:
            ip_d = self._get_ip_day_bucket(ip)
            ip_d["active_sec"] += duration_sec

            g_d = self._get_global_day()
            g_d["active_sec"] += duration_sec
            g_d["cost"] += cost

            g_m_cost = self._get_global_month_cost()
            g_m_cost["cost"] += cost
