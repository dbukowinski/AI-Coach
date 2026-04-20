from __future__ import annotations

# Pipeline Strava → analiza → briefing (funkcje dla Streamlit).
# UI i dialog: app.py + session_state. NIE ustawiaj tego pliku jako Main file na Cloud.

from agent_state import AgentState
from agent_nodes import (
    analyze_training_node,
    clean_data_node,
    coaching_brief_node,
    ensure_token_node,
    fetch_activities_node,
    fetch_new_details_node,
    report_node,
    save_raw_node,
    streamlit_seed_coaching_messages,
)

PIPELINE_TO_COACHING_BRIEF = [
    ensure_token_node,
    fetch_activities_node,
    fetch_new_details_node,
    save_raw_node,
    clean_data_node,
    report_node,
    analyze_training_node,
    coaching_brief_node,
]


def run_sync_pipeline_to_coaching_brief(state: AgentState) -> AgentState:
    """
    Wykonuje ten sam łańcuch co graf LangGraph do momentu `coaching_brief_node`,
    potem wstawia pierwszą wiadomość coacha do `state.messages`.
    """
    for fn in PIPELINE_TO_COACHING_BRIEF:
        state = fn(state)
        if state.done or state.errors:
            return state
    return streamlit_seed_coaching_messages(state)


if __name__ == "__main__":
    # Ktoś uruchomił `streamlit run streamlit_runner.py` — to nie jest aplikacja UI.
    import streamlit as st

    st.set_page_config(page_title="AI Coach — zła konfiguracja", layout="centered")
    st.error(
        "**Ten plik nie jest aplikacją Streamlit.**\n\n"
        "W *Settings → Main file path* ustaw **`streamlit_app.py`** albo **`app.py`**.\n\n"
        "`streamlit_runner.py` to tylko funkcje pomocnicze (import z `app.py`)."
    )
    st.markdown(
        "[Dokumentacja Streamlit — punkt wejścia (entrypoint)](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app#entrypoint)"
    )
