from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent_state import AgentState
from agent_nodes import (
    analyze_training_node,
    ask_user_node,
    clean_data_node,
    display_plan_node,
    ensure_token_node,
    fetch_activities_node,
    fetch_new_details_node,
    finish_node,
    generate_plan_node,
    report_node,
    revise_plan_node,
    route_after_think,
    save_raw_node,
    think_node,
)


def build_graph():
    graph = StateGraph(AgentState)

    # deterministic ingest/report
    graph.add_node("ensure_token", ensure_token_node)
    graph.add_node("fetch_activities", fetch_activities_node)
    graph.add_node("fetch_new_details", fetch_new_details_node)
    graph.add_node("save_raw", save_raw_node)
    graph.add_node("clean_data", clean_data_node)
    graph.add_node("report", report_node)

    # reasoning loop
    graph.add_node("think", think_node)
    graph.add_node("analyze_training", analyze_training_node)
    graph.add_node("generate_plan", generate_plan_node)
    graph.add_node("display_plan", display_plan_node)
    graph.add_node("ask_user", ask_user_node)
    graph.add_node("revise_plan", revise_plan_node)
    graph.add_node("finish", finish_node)

    # entry routing by mode
    def route_start(state: AgentState) -> str:
        if state.mode == "analysis":
            return "analysis_path"
        return "full_path"

    graph.add_conditional_edges(
        START,
        route_start,
        {
            "full_path": "ensure_token",
            "analysis_path": "think",
        },
    )

    # deterministic full path
    graph.add_edge("ensure_token", "fetch_activities")
    graph.add_edge("fetch_activities", "fetch_new_details")
    graph.add_edge("fetch_new_details", "save_raw")
    graph.add_edge("save_raw", "clean_data")
    graph.add_edge("clean_data", "report")
    graph.add_edge("report", "think")

    # reasoning loop
    graph.add_conditional_edges(
        "think",
        route_after_think,
        {
            "analyze_training": "analyze_training",
            "generate_plan": "generate_plan",
            "ask_user": "ask_user",
            "revise_plan": "revise_plan",
            "finish": "finish",
        },
    )

    graph.add_edge("analyze_training", "think")
    graph.add_edge("generate_plan", "display_plan")
    graph.add_edge("display_plan", "think")
    graph.add_edge("ask_user", "think")
    graph.add_edge("revise_plan", "display_plan")
    graph.add_edge("finish", END)

    return graph.compile()