from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agent_state import AgentState
from agent_nodes import (
    analyze_training_node,
    ask_user_node,
    clean_data_node,
    coach_dialog_node,
    coaching_brief_node,
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
from intent_classifier import (
    intent_classifier_node,
    route_after_intent,
    handle_question_node,
    handle_training_plan_node,
    handle_route_node,
    handle_nutrition_node,
    handle_race_node,
    handle_context_info_node,
    handle_gear_node,
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
    graph.add_node("coaching_brief", coaching_brief_node)
    graph.add_node("coach_dialog", coach_dialog_node)
    graph.add_node("generate_plan", generate_plan_node)
    graph.add_node("display_plan", display_plan_node)
    graph.add_node("ask_user", ask_user_node)
    graph.add_node("revise_plan", revise_plan_node)
    graph.add_node("finish", finish_node)

    # intent classification + 7 handler branches
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("handle_question", handle_question_node)
    graph.add_node("handle_training_plan", handle_training_plan_node)
    graph.add_node("handle_route", handle_route_node)
    graph.add_node("handle_nutrition", handle_nutrition_node)
    graph.add_node("handle_race", handle_race_node)
    graph.add_node("handle_context_info", handle_context_info_node)
    graph.add_node("handle_gear", handle_gear_node)

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
            "coaching_brief": "coaching_brief",
            "coach_dialog": "coach_dialog",
            "generate_plan": "generate_plan",
            "ask_user": "ask_user",
            "revise_plan": "revise_plan",
            "finish": "finish",
        },
    )

    graph.add_edge("analyze_training", "think")
    graph.add_edge("coaching_brief", "think")
    graph.add_edge("coach_dialog", "think")
    graph.add_edge("generate_plan", "display_plan")
    graph.add_edge("display_plan", "think")
    graph.add_edge("revise_plan", "display_plan")
    graph.add_edge("finish", END)

    # ask_user → intent_classifier when feedback present, else back to think
    def route_after_ask_user(state: AgentState) -> str:
        if state.done or state.plan_accepted or not state.user_feedback.strip():
            return "think"
        return "intent_classifier"

    graph.add_conditional_edges(
        "ask_user",
        route_after_ask_user,
        {
            "think": "think",
            "intent_classifier": "intent_classifier",
        },
    )

    # intent_classifier → one of 7 handler branches
    graph.add_conditional_edges(
        "intent_classifier",
        route_after_intent,
        {
            "handle_question": "handle_question",
            "handle_training_plan": "handle_training_plan",
            "handle_route": "handle_route",
            "handle_nutrition": "handle_nutrition",
            "handle_race": "handle_race",
            "handle_context_info": "handle_context_info",
            "handle_gear": "handle_gear",
        },
    )

    # all handler branches return to think to continue the reasoning loop
    graph.add_edge("handle_question", "think")
    graph.add_edge("handle_training_plan", "think")
    graph.add_edge("handle_route", "think")
    graph.add_edge("handle_nutrition", "think")
    graph.add_edge("handle_race", "think")
    graph.add_edge("handle_context_info", "think")
    graph.add_edge("handle_gear", "think")

    return graph.compile()
