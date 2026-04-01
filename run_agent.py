from __future__ import annotations

import argparse

from agent_graph import build_graph
from agent_state import AgentState


def parse_args():
    parser = argparse.ArgumentParser(description="AI Coach Strava Agent")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "analysis"],
        help="Run full pipeline or analysis-only mode",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=12,
        help="Safety limit for reasoning loop (more = more feedback rounds)",
    )
    parser.add_argument(
        "--max-weekly-minutes",
        type=int,
        default=600,
        help="Soft cap on planned training minutes per week",
    )
    parser.add_argument(
        "--day-off",
        action="append",
        default=[],
        help="Day without training (e.g. monday). Can be used multiple times.",
    )
    parser.add_argument(
        "--quality-day",
        action="append",
        default=[],
        help="Preferred day for quality / harder sessions (e.g. tuesday). Can be used multiple times.",
    )
    parser.add_argument(
        "--max-sessions-per-day",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Max sessions per day (2–3 = split into morning/midday/evening, balanced RPE).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Use CLI-based HITL instead of Tkinter window",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    app = build_graph()

    initial_state = AgentState(
        days=args.days,
        mode=args.mode,
        hitl_mode="cli" if args.no_gui else "gui",
        max_weekly_minutes=args.max_weekly_minutes,
        days_off=args.day_off or [],
        preferred_quality_days=args.quality_day or [],
        max_sessions_per_day=args.max_sessions_per_day,
        max_loops=args.max_loops,
    )

    final_state = app.invoke(
        initial_state,
        config={"recursion_limit": 50},
    )

    print("\n=== FINAL STATE ===")

    if isinstance(final_state, dict):
        print(f"done: {final_state.get('done')}")
        print(f"plan_accepted: {final_state.get('plan_accepted')}")
        print(f"errors: {final_state.get('errors')}")
        print(f"next_action: {final_state.get('next_action')}")
        print(f"loop_count: {final_state.get('loop_count')}")
    else:
        print(f"done: {final_state.done}")
        print(f"plan_accepted: {final_state.plan_accepted}")
        print(f"errors: {final_state.errors}")
        print(f"next_action: {final_state.next_action}")
        print(f"loop_count: {final_state.loop_count}")


if __name__ == "__main__":
    main()