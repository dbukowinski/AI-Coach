"""
AI Coach — Streamlit UI z prawdziwym pipeline (Strava → analiza → dialog coachingowy → plan).

Uruchomienie z katalogu projektu:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from agent_state import AgentState
from agent_nodes import (
    build_streamlit_demo_agent_state,
    streamlit_coach_after_user,
    streamlit_finalize_coaching_and_plan,
)
from streamlit_runner import run_sync_pipeline_to_coaching_brief

# ── Konfiguracja strony ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Coach",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0; max-width: 100%; }
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
        "Bez demo wymagane są token Strava / AWS (Bedrock) zgodnie z README projektu."
    )

# ── Nagłówek ──────────────────────────────────────────────────────────────────

st.markdown("### 🏃 AI Coach")
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
        disabled=not chat_ready or plan_done,
    ):
        if agent:
            with st.spinner("Coach myśli…"):
                agent, plan_ready = streamlit_coach_after_user(agent, prompt)
                if plan_ready:
                    agent = streamlit_finalize_coaching_and_plan(agent)
                st.session_state.agent = agent
        st.rerun()

    if plan_done:
        st.success("Plan jest po prawej — możesz odświeżyć briefing, aby zacząć od nowa.")

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
