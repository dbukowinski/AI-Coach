"""
AI Coach - Streamlit UI
Chat po lewej, plan tygodniowy po prawej.
Wymaga: streamlit, langchain/bedrock (lub mock), sync_strava
"""

import streamlit as st
from datetime import date

# ── Konfiguracja strony ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Coach",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS: minimalizuje domyślny padding Streamlit ─────────────────────────────

st.markdown("""
<style>
    /* usuń domyślny padding strony */
    .block-container { padding-top: 1rem; padding-bottom: 0; max-width: 100%; }

    /* chat input na dole kolumny */
    [data-testid="stChatInput"] { border-top: 1px solid #eee; }

    /* metryki — mniejszy font label */
    [data-testid="stMetricLabel"] { font-size: 12px !important; }

    /* dzień odpoczynku — wyszarzony */
    .rest-day { opacity: 0.45; }

    /* badge "zmieniono" */
    .badge-changed {
        background: #FAEEDA;
        color: #633806;
        font-size: 10px;
        padding: 2px 7px;
        border-radius: 999px;
        margin-left: 6px;
    }

    /* komentarz coachingowy */
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
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "messages": [],           # historia czatu: [{role, content}]
        "plan": None,             # wygenerowany plan tygodniowy
        "dialog_complete": False, # coach zdecydował że ma dość kontekstu
        "training_summary": None, # dane z analysis.py
        "strava_connected": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Mock danych (zastąp sync_strava + analysis) ───────────────────────────────

MOCK_SUMMARY = {
    "last_28d_km": 187,
    "last_28d_runs": 18,
    "last_week_km": 22,      # lżejszy tydzień
    "avg_weekly_km": 47,
    "risk_flags": ["spadek objętości ostatni tydzień (-53%)"],
    "patterns": ["biegasz głównie rano", "wolne w weekendy są dłuższe"],
}

MOCK_PLAN = {
    "weekly_km": 38,
    "run_days": 4,
    "intensity": "niska",
    "coach_note": (
        "Ten tydzień to powrót do rytmu — nie buduj objętości. "
        "Jeśli w piątek poczujesz się świetnie, nie dodawaj sesji. "
        "Regeneracja jest częścią planu."
    ),
    "days": [
        {"name": "Poniedziałek", "type": "easy",  "km": 8,  "note": "Łatwy bieg, strefa 2. Powrót do rytmu."},
        {"name": "Wtorek",       "type": "rest",  "km": 0,  "note": "Regeneracja aktywna lub wolne."},
        {"name": "Środa",        "type": "easy",  "km": 6,  "note": "Lekki bieg, strefa 1-2.", "changed": True},
        {"name": "Czwartek",     "type": "rest",  "km": 0,  "note": "Wolne."},
        {"name": "Piątek",       "type": "rest",  "km": 0,  "note": "Nogi przed weekendem."},
        {"name": "Sobota",       "type": "easy",  "km": 10, "note": "Długi bieg, spokojne tempo."},
        {"name": "Niedziela",    "type": "easy",  "km": 14, "note": "Wybieganie, wolne tempo."},
    ],
}

TYPE_COLORS = {
    "easy":   "#5DCAA5",
    "tempo":  "#EF9F27",
    "long":   "#1D9E75",
    "rest":   "#D3D1C7",
    "race":   "#E24B4A",
}


# ── Nagłówek ──────────────────────────────────────────────────────────────────

header_col, status_col = st.columns([3, 1])
with header_col:
    st.markdown("### 🏃 AI Coach")
with status_col:
    if st.session_state.strava_connected:
        st.success("Strava połączona", icon="✓")
    else:
        if st.button("Połącz Strava", use_container_width=True):
            # TODO: uruchom OAuth flow
            st.session_state.strava_connected = True
            st.session_state.training_summary = MOCK_SUMMARY
            st.rerun()

st.divider()


# ── Pasek kontekstu (dane z ostatnich 28 dni) ────────────────────────────────

if st.session_state.training_summary:
    s = st.session_state.training_summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ostatnie 28 dni", f"{s['last_28d_km']} km")
    c2.metric("Liczba biegów", s['last_28d_runs'])
    c3.metric("Ostatni tydzień", f"{s['last_week_km']} km",
              delta=f"{s['last_week_km'] - s['avg_weekly_km']} km vs średnia",
              delta_color="inverse")
    with c4:
        if s["risk_flags"]:
            st.warning(s["risk_flags"][0], icon="⚠️")
    st.divider()


# ── Dwie kolumny: CHAT | PLAN ─────────────────────────────────────────────────

chat_col, plan_col = st.columns([1, 1], gap="medium")


# ── LEWA: Chat ────────────────────────────────────────────────────────────────

with chat_col:
    st.markdown("**Rozmowa z trenerem**")

    # kontener na wiadomości — scrollowalny
    chat_container = st.container(height=480)

    with chat_container:
        # pierwsze otwarcie: coach zaczyna od danych, nie od pytania
        if not st.session_state.messages and st.session_state.training_summary:
            opening = (
                "Widzę że w ostatnich 4 tygodniach biegałeś regularnie — "
                "4-5 razy tygodniowo. Ale ostatni tydzień był wyraźnie lżejszy: "
                "tylko 2 biegi i oba krótkie. Coś się działo?"
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": opening}
            )

        # renderuj historię
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"],
                                 avatar="🏃" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

    # input użytkownika
    if user_input := st.chat_input("Napisz do trenera...",
                                   disabled=not st.session_state.strava_connected):
        # dodaj wiadomość użytkownika
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        # ── wywołanie agenta ──────────────────────────────────────────────────
        # Tutaj podpinasz node_coach_turn() z agent_nodes.py
        # Poniżej mock odpowiedzi — zastąp wywołaniem LangGraph

        with st.spinner("Coach myśli..."):
            # MOCK: zastąp agent_graph.invoke(state)
            if "półmaraton" in user_input.lower() or "maraton" in user_input.lower():
                reply = (
                    "7 tygodni to dobry horyzont — wystarczy na blok budowania i taper. "
                    "Proponuję umiarkowany tydzień: 4 biegi, bez intensywności. "
                    "Wygenerowałem plan po prawej. Co chcesz zmienić?"
                )
                st.session_state.plan = MOCK_PLAN
            elif any(w in user_input.lower() for w in ["środa", "czwartek", "nie mogę", "zmień"]):
                reply = (
                    "Bez problemu — zaktualizowałem plan. "
                    "Środa stała się dniem wolnym, czwartek przejął lekki bieg."
                )
                # aktualizuj plan inline
                if st.session_state.plan:
                    for day in st.session_state.plan["days"]:
                        if day["name"] == "Środa":
                            day["type"] = "rest"
                            day["km"] = 0
                            day["changed"] = True
                        if day["name"] == "Czwartek":
                            day["type"] = "easy"
                            day["km"] = 6
                            day["note"] = "Lekki bieg, przeniesiony ze środy."
            else:
                reply = (
                    "Rozumiem. Na tej podstawie dostosowuję podejście. "
                    "Czy chcesz żebym wygenerował plan na ten tydzień?"
                )

            st.session_state.messages.append(
                {"role": "assistant", "content": reply}
            )

        st.rerun()

    # hint gdy Strava nie połączona
    if not st.session_state.strava_connected:
        st.info("Połącz Strava żeby rozpocząć rozmowę z trenerem.")


# ── PRAWA: Plan tygodniowy ────────────────────────────────────────────────────

with plan_col:
    st.markdown("**Plan tygodniowy**")

    if not st.session_state.plan:
        st.markdown(
            "<div style='color: #888; font-size: 14px; margin-top: 40px; text-align: center;'>"
            "Plan pojawi się tutaj po rozmowie z trenerem."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        plan = st.session_state.plan

        # metryki podsumowujące
        m1, m2, m3 = st.columns(3)
        m1.metric("Objętość", f"{plan['weekly_km']} km")
        m2.metric("Biegi", f"{plan['run_days']} dni")
        m3.metric("Intensywność", plan["intensity"])

        st.markdown("---")

        # dni tygodnia
        for day in plan["days"]:
            color = TYPE_COLORS.get(day["type"], "#ccc")
            is_rest = day["type"] == "rest"
            changed = day.get("changed", False)

            # nagłówek dnia
            day_header = f"**{day['name']}**"
            if changed:
                day_header += " <span class='badge-changed'>zmieniono</span>"
            if not is_rest:
                day_header += f"&nbsp;&nbsp;<span style='color:#888;font-size:12px;'>{day['km']} km</span>"

            with st.container():
                col_dot, col_content = st.columns([0.04, 0.96])
                with col_dot:
                    st.markdown(
                        f"<div style='width:8px;height:8px;border-radius:50%;"
                        f"background:{color};margin-top:6px;'></div>",
                        unsafe_allow_html=True,
                    )
                with col_content:
                    if is_rest:
                        st.markdown(
                            f"<div class='rest-day'>{day_header}<br>"
                            f"<span style='font-size:12px;color:#aaa;'>{day['note']}</span></div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"{day_header}<br>"
                            f"<span style='font-size:12px;color:#555;'>{day['note']}</span>",
                            unsafe_allow_html=True,
                        )

            st.markdown(
                "<hr style='margin:4px 0;border:none;border-top:0.5px solid #eee;'>",
                unsafe_allow_html=True,
            )

        # komentarz coachingowy
        st.markdown(
            f"<div class='coach-note'>{plan['coach_note']}</div>",
            unsafe_allow_html=True,
        )

        # eksport (placeholder)
        st.markdown("<br>", unsafe_allow_html=True)
        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.button("Eksportuj do CSV", use_container_width=True, disabled=True)
        with export_col2:
            st.button("Dodaj do kalendarza", use_container_width=True, disabled=True)
