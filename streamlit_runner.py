from __future__ import annotations

"""
Uruchomienie pipeline Strava → analiza → briefing coachingowy dla Streamlit.
Pełny dialog i generowanie planu odbywają się w app.py (session_state + wywołania z agent_nodes).
"""

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
