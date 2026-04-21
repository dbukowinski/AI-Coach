"""
AI Coach — Streamlit UI z prawdziwym pipeline (Strava → analiza → dialog coachingowy → plan).

Uruchomienie z katalogu projektu:
    streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from agent_state import AgentState
from agent_nodes import (
    build_streamlit_demo_agent_state,
    streamlit_coach_after_user,
    streamlit_finalize_coaching_and_plan,
    streamlit_revise_plan_after_user,
)
from streamlit_runner import run_sync_pipeline_to_coaching_brief

# ── Konfiguracja strony ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Coach",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0; }
    [data-testid="stChatInput"] { border-top: 1px solid #eee; }
    [data-testid="stMetricLabel"] { font-size: 12px !important; }
    .rest-day { opacity: 0.45; }
    .badge-changed {
        background: #FAEEDA;
        color: #633806;
        font-size: 10px;
        padding: 2px 7px;
        border-radius: 999px;
        margin-left: 6px;
    }
    .coach-note {
        background: #E1F5EE;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #085041;
        line-height: 1.55;
        margin-top: 12px;
    }

    /* tytuł aplikacji — unikaj ucięcia emoji/fontów */
    .app-title {
        font-size: 30px;
        font-weight: 800;
        line-height: 1.35;
        padding: 0.25rem 0 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

PL_WEEKDAYS = [
    "Poniedziałek",
    "Wtorek",
    "Środa",
    "Czwartek",
    "Piątek",
    "Sobota",
    "Niedziela",
]

TYPE_COLORS = {
    "easy": "#5DCAA5",
    "tempo": "#EF9F27",
    "long": "#1D9E75",
    "rest": "#D3D1C7",
    "race": "#E24B4A",
    "other": "#888888",
}


def _init_session() -> None:
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "use_demo" not in st.session_state:
        st.session_state.use_demo = False


def _metrics_from_agent(agent: AgentState) -> Dict[str, Any]:
    w = agent.weekly_summary or {}
    fw = agent.four_week_summary or {}
    cur = float(w.get("weekly_load") or 0)
    avg = float(fw.get("avg_4w_load") or 0)
    delta = (cur - avg) if avg else None
    return {
        "activity_count": w.get("activity_count", "—"),
        "weekly_load": round(cur, 2) if cur else None,
        "avg_4w_load": round(avg, 2) if avg else None,
        "delta": delta,
        "flags": list(agent.flags or []),
        "patterns": list(agent.patterns or []),
    }


def _intensity_bucket(session: Dict[str, Any]) -> str:
    intensity = str(session.get("intensity", "")).lower()
    stype = str(session.get("session_type", "")).lower()
    if intensity == "rest" or "rest day" in stype:
        return "rest"
    if intensity in ("long",):
        return "long"
    if intensity in ("hard", "moderate-hard", "moderate", "high"):
        return "tempo"
    if intensity in ("easy", "recovery"):
        return "easy"
    return "other"


def _weekday_label(date_iso: str) -> str:
    try:
        wd = datetime.fromisoformat(date_iso).weekday()
        return PL_WEEKDAYS[wd]
    except Exception:
        return date_iso or "—"

def _get_secret(key: str) -> Optional[str]:
    try:
        if key in st.secrets and st.secrets[key] not in (None, ""):
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return None


def _exchange_strava_code_for_tokens(client_id: str, client_secret: str, code: str) -> Dict[str, Any]:
    import requests

    r = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Strava token exchange failed: {r.status_code} {r.text}")
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError("Strava token exchange returned non-JSON object")
    return data


def _has_strava_connection() -> bool:
    # 1) env (z secrets lub runtime)
    if (os.getenv("STRAVA_REFRESH_TOKEN") or "").strip():
        return True
    # 2) local store (data/strava_tokens.json)
    try:
        from strava_token_store import get_active_tokens

        t = get_active_tokens()
        if not t:
            return False
        return bool(str(t.get("refresh_token") or "").strip())
    except Exception:
        return False


def _render_plan(plan: Dict[str, Any]) -> None:
    sessions: List[Dict[str, Any]] = plan.get("sessions") or []
    template = plan.get("template", "—")
    explanation = (plan.get("explanation") or "").strip()

    total_min = sum(int(s.get("duration_min") or 0) for s in sessions)
    run_days = sum(
        1
        for s in sessions
        if int(s.get("duration_min") or 0) > 0
        and str(s.get("intensity", "")).lower() != "rest"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Łączny czas (plan)", f"{total_min} min")
    m2.metric("Dni z treningiem", run_days)
    m3.metric("Szablon", str(template))

    if explanation:
        st.markdown(f'<div class="coach-note">{explanation}</div>', unsafe_allow_html=True)

    st.markdown("---")

    for s in sessions:
        day_label = _weekday_label(str(s.get("date", "")))
        stype = str(s.get("session_type", "—"))
        duration = str(s.get("duration", "—"))
        intensity = str(s.get("intensity", "—"))
        notes = str(s.get("notes", "") or "")
        bucket = _intensity_bucket(s)
        color = TYPE_COLORS.get(bucket, TYPE_COLORS["other"])
        is_rest = bucket == "rest"

        with st.container():
            col_dot, col_content = st.columns([0.04, 0.96])
            with col_dot:
                st.markdown(
                    f'<div style="width:8px;height:8px;border-radius:50%;background:{color};margin-top:6px;"></div>',
                    unsafe_allow_html=True,
                )
            with col_content:
                header = f"**{day_label}** ({s.get('date', '')}) — {stype}"
                if is_rest:
                    st.markdown(
                        f'<div class="rest-day">{header}<br>'
                        f'<span style="font-size:12px;color:#aaa;">{duration} · {intensity}</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"{header}<br>"
                        f'<span style="font-size:12px;color:#555;">{duration} · {intensity}</span>',
                        unsafe_allow_html=True,
                    )
                if notes:
                    st.caption(notes)
        st.markdown(
            '<hr style="margin:4px 0;border:none;border-top:0.5px solid #eee;">',
            unsafe_allow_html=True,
        )


_init_session()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Ustawienia")
    days = st.number_input("Okno dni (Strava / analiza)", min_value=7, max_value=90, value=28, step=1)
    use_demo = st.toggle("Tryb demo (bez Strava, przykładowe metryki)", value=st.session_state.use_demo)
    if use_demo != st.session_state.use_demo:
        st.session_state.use_demo = use_demo
        st.session_state.agent = None
        st.rerun()

    st.divider()
    if st.button("Przygotuj dane i briefing", type="primary", use_container_width=True):
        with st.spinner("Synchronizacja i analiza…"):
            if use_demo:
                st.session_state.agent = build_streamlit_demo_agent_state()
            else:
                if not _has_strava_connection():
                    st.error(
                        "Najpierw połącz Stravę: otwórz sekcję „Połącz Stravę (automatyczny OAuth…)”, "
                        "kliknij link i zaakceptuj. Dopiero potem uruchom pipeline."
                    )
                    st.stop()
                agent = AgentState(
                    days=int(days),
                    mode="full",
                    hitl_mode="streamlit",
                )
                st.session_state.agent = run_sync_pipeline_to_coaching_brief(agent)
        if st.session_state.agent and st.session_state.agent.errors:
            for e in st.session_state.agent.errors:
                st.error(e)
        elif st.session_state.agent:
            st.success("Gotowe — możesz pisać z trenerem po lewej.")
        st.rerun()

    st.caption(
        "Bez demo: Strava + (opcjonalnie) AWS Bedrock. Lokalnie: plik `.env`. "
        "Na Streamlit Cloud: **Settings → Secrets** (nie commituj sekretów)."
    )
    with st.expander("Jak dodać Stravę (prosto i jasno)"):
        st.markdown(
            "**Co musisz zrobić TY (właściciel apki) – tylko raz:**\n"
            "1. Wejdź w `https://www.strava.com/settings/api` i kliknij **Create Application**.\n"
            "2. Skopiuj z tej strony:\n"
            "   - **Client ID** → `STRAVA_CLIENT_ID`\n"
            "   - **Client Secret** → `STRAVA_CLIENT_SECRET`\n"
            "3. W tym samym formularzu Stravy ustaw:\n"
            "   - **Authorization Callback Domain** = domena Twojej apki (np. `zabiegany.streamlit.app`).\n"
            "4. W Streamlit Cloud wejdź w **Manage app → Settings → Secrets** i wklej:\n"
        )
        st.code(
            'STRAVA_CLIENT_ID = "12345"  # skopiuj: Strava → Settings → API → Client ID\n'
            'STRAVA_CLIENT_SECRET = "a1b2c3d4e5f6..."  # skopiuj: Client Secret\n'
            'APP_URL = "https://zabiegany.streamlit.app"  # URL z paska przeglądarki\n',
            language="toml",
        )
        st.markdown(
            "**Wzór domeny do Stravy:**\n"
            "- Jeśli Twój URL to `https://zabiegany.streamlit.app` → w Stravie wpisz **tylko**: `zabiegany.streamlit.app`\n"
            "- Bez `https://` i bez ścieżek.\n"
        )
        st.markdown(
            "**Co robi użytkownik (biegacz):**\n"
            "- Otwiera sekcję **„Połącz Stravę (automatyczny OAuth…)”**, klika link, loguje się i akceptuje.\n"
            "- Tokeny zapisują się automatycznie po stronie serwera (w `data/`), więc nic nie wkleja ręcznie."
        )
        st.markdown("Na koniec kliknij **Reboot app** (Streamlit Cloud) po zmianie Secrets.")

    with st.expander("Połącz Stravę (automatyczny OAuth w aplikacji)"):
        st.markdown(
            "Ta sekcja **automatyzuje uzyskanie** `STRAVA_REFRESH_TOKEN` w przeglądarce. "
            "Tokeny zostaną zapisane po stronie serwera (w `data/`, gitignored), "
            "więc użytkownik nie musi wklejać ich do Secrets."
        )

        client_id = _get_secret("STRAVA_CLIENT_ID") or (os.getenv("STRAVA_CLIENT_ID") or "").strip()
        client_secret = _get_secret("STRAVA_CLIENT_SECRET") or (os.getenv("STRAVA_CLIENT_SECRET") or "").strip()
        app_url = (
            _get_secret("APP_URL")
            or _get_secret("STREAMLIT_APP_URL")
            or (os.getenv("APP_URL") or os.getenv("STREAMLIT_APP_URL") or "").strip()
        )

        if not client_id or not client_secret:
            st.warning(
                "Najpierw dodaj w Secrets: `STRAVA_CLIENT_ID` i `STRAVA_CLIENT_SECRET` "
                "(z `https://www.strava.com/settings/api`)."
            )
        else:
            if not app_url:
                st.info(
                    "Ustaw w Secrets `APP_URL` (pełny adres aplikacji), np. "
                    "`https://twoja-app.streamlit.app` — użyjemy go jako `redirect_uri`."
                )
            else:
                redirect_uri = app_url.rstrip("/") + "/"
                authorize_url = (
                    "https://www.strava.com/oauth/authorize"
                    f"?client_id={client_id}"
                    "&response_type=code"
                    f"&redirect_uri={redirect_uri}"
                    "&approval_prompt=force"
                    "&scope=read,activity:read_all"
                )
                st.markdown(f"[Kliknij aby połączyć Stravę]({authorize_url})")
                st.caption(
                    "Po akceptacji Strava wróci do aplikacji z parametrem `code` w URL."
                )

                # Jeśli Strava wróciła z code=..., wymień na tokeny i pokaż instrukcję.
                code = st.query_params.get("code")
                if isinstance(code, list):
                    code = code[0] if code else None
                if code:
                    try:
                        tokens = _exchange_strava_code_for_tokens(client_id, client_secret, str(code))
                        refresh_token = tokens.get("refresh_token", "")
                        access_token = tokens.get("access_token", "")
                        expires_at = tokens.get("expires_at", 0)
                        athlete_id = None
                        try:
                            athlete_id = (tokens.get("athlete") or {}).get("id")
                        except Exception:
                            athlete_id = None
                        st.success("Połączenie OK — odebrałem tokeny ze Stravy.")
                        # Ułatwienie: ustaw w bieżącym procesie, żeby pipeline zadziałał od razu.
                        os.environ["STRAVA_REFRESH_TOKEN"] = str(refresh_token)
                        os.environ["STRAVA_ACCESS_TOKEN"] = str(access_token)
                        os.environ["STRAVA_EXPIRES_AT"] = str(expires_at)

                        # Persist dla kolejnych wejść (bez Secrets)
                        try:
                            from strava_token_store import upsert_tokens

                            upsert_tokens(
                                athlete_id or "default",
                                access_token=str(access_token),
                                refresh_token=str(refresh_token),
                                expires_at=int(expires_at or 0),
                            )
                        except Exception as e:
                            st.warning("Nie udało się zapisać tokenów lokalnie (zostaną tylko na tę sesję).")
                            st.code(str(e))

                        st.info("Możesz teraz kliknąć „Przygotuj dane i briefing” — bez wklejania sekretów.")
                    except Exception as e:
                        st.error("Nie udało się wymienić `code` na tokeny.")
                        st.code(str(e))

    with st.expander("Diagnostyka Stravy (Cloud)"):
        import os

        def _mask(v: object) -> str:
            if v is None:
                return "brak"
            s = str(v).strip()
            if not s:
                return "puste"
            if len(s) <= 6:
                return "***"
            return s[:2] + "…" + s[-2:]

        keys = [
            "STRAVA_CLIENT_ID",
            "STRAVA_CLIENT_SECRET",
            "STRAVA_REFRESH_TOKEN",
            "STRAVA_ACCESS_TOKEN",
            "STRAVA_EXPIRES_AT",
        ]
        st.markdown("**Wykryte wartości (maskowane):**")
        for k in keys:
            st.write(f"- `{k}`: `{_mask(os.getenv(k))}`")

        if st.button("Test: odśwież token + pobierz 1 aktywność", use_container_width=True):
            try:
                import sync_strava

                sync_strava.ensure_valid_token()
                acts = sync_strava.fetch_last_days(days=1, per_page=1)
                st.success(f"OK — token działa. Pobrano {len(acts)} aktywność(‑i) z ostatniego dnia.")
            except Exception as e:
                st.error("Strava test nie powiódł się.")
                import traceback

                st.markdown("**Szczegóły błędu:**")
                st.code(repr(e))
                st.markdown("**Traceback:**")
                st.code(traceback.format_exc())

# ── Nagłówek ──────────────────────────────────────────────────────────────────

st.markdown('<div class="app-title">🏃 AI Coach</div>', unsafe_allow_html=True)
agent: Optional[AgentState] = st.session_state.agent

if agent and not agent.errors and agent.coaching_brief_ready:
    meta = _metrics_from_agent(agent)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aktywności (okno)", meta["activity_count"])
    if meta["weekly_load"] is not None:
        c2.metric("Obciążenie (tydzień)", meta["weekly_load"])
    else:
        c2.metric("Obciążenie (tydzień)", "—")
    if meta["avg_4w_load"] is not None:
        c3.metric("Średnia 4 tyg.", meta["avg_4w_load"])
    else:
        c3.metric("Średnia 4 tyg.", "—")
    if meta["delta"] is not None:
        c4.metric("Δ vs średnia 4t", round(meta["delta"], 2))
    else:
        c4.metric("Δ vs średnia 4t", "—")

    if meta["flags"]:
        st.warning(" · ".join(meta["flags"]), icon="⚠️")
    if meta["patterns"]:
        with st.expander("Wzorce z briefingu"):
            for p in meta["patterns"]:
                st.markdown(f"- {p}")

st.divider()

chat_col, plan_col = st.columns([1, 1], gap="medium")

# ── Chat ──────────────────────────────────────────────────────────────────────

with chat_col:
    st.markdown("**Rozmowa z trenerem**")
    chat_ready = bool(
        agent
        and not agent.errors
        and agent.coaching_brief_ready
        and agent.messages
    )
    plan_done = bool(agent and agent.plan_draft)

    chat_container = st.container(height=480)
    with chat_container:
        if agent and agent.messages:
            for msg in agent.messages:
                role = msg.get("role", "user")
                with st.chat_message(role, avatar="🏃" if role == "assistant" else "👤"):
                    st.markdown(msg.get("content", ""))
        elif not agent:
            st.info('Włącz demo lub kliknij „Przygotuj dane i briefing” w panelu bocznym.')
        elif agent.errors:
            st.error("Napraw błędy pipeline (logi powyżej) i spróbuj ponownie.")

    if prompt := st.chat_input(
        "Napisz do trenera…",
        disabled=not chat_ready,
    ):
        if agent:
            with st.spinner("Coach myśli…"):
                # Przed planem: dialog coachingowy -> PLAN_READY -> generowanie planu
                if not agent.plan_draft:
                    agent, plan_ready = streamlit_coach_after_user(agent, prompt)
                    if plan_ready:
                        agent = streamlit_finalize_coaching_and_plan(agent)
                # Po planie: kolejne wiadomości rewizują plan
                else:
                    agent = streamlit_revise_plan_after_user(agent, prompt)
                st.session_state.agent = agent
        st.rerun()

    if plan_done:
        st.info("Plan jest po prawej — możesz pisać dalej, a ja będę go aktualizował na podstawie Twoich wiadomości.")

# ── Plan ──────────────────────────────────────────────────────────────────────

with plan_col:
    st.markdown("**Plan tygodniowy**")
    if agent and agent.plan_draft:
        _render_plan(agent.plan_draft)
    elif agent and agent.errors and not agent.plan_draft:
        st.error("Nie udało się zbudować planu.")
        for err in agent.errors:
            st.code(err)
    else:
        st.markdown(
            "<div style='color: #888; font-size: 14px; margin-top: 40px; text-align: center;'>"
            "Plan pojawi się po zakończeniu rozmowy (gdy trener uzna, że ma wystarczający kontekst "
            "— w odpowiedzi używa sygnału wewnętrznego PLAN_READY)."
            "</div>",
            unsafe_allow_html=True,
        )
