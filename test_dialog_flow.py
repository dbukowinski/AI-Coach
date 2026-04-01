from __future__ import annotations

import sys

from agent_nodes import revise_plan_node
from agent_state import AgentState
from plan_generator import generate_training_plan


def run_case(label: str, state: AgentState, feedback: str) -> None:
    print(f"\n=== CASE: {label} ===")
    state.user_feedback = feedback
    state = revise_plan_node(state)
    print(f"coach_question: {state.coach_question!r}")
    print(f"max_training_days: {state.max_training_days!r}")
    print(f"weekend_focus: {state.weekend_focus!r}")
    print(f"days_off: {state.days_off!r}")
    print(f"preferred_quality_days: {state.preferred_quality_days!r}")
    print(f"max_sessions_per_day: {state.max_sessions_per_day!r}")
    print(f"sessions: {len((state.plan_draft or {}).get('sessions', []))}")
    # show first day sessions
    sessions = (state.plan_draft or {}).get("sessions", [])
    if sessions:
        print("first sessions:")
        for s in sessions[: min(5, len(sessions))]:
            print(f"- {s.get('date')} | {s.get('title', s.get('session_type'))} | {s.get('duration')} | {s.get('intensity')}")


def main() -> None:
    # ensure utf-8 output on Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Minimal state with an initial plan draft (no Strava needed)
    plan = generate_training_plan(
        days=7,
        max_weekly_minutes=600,
        days_off=[],
        preferred_quality_days=[],
        max_sessions_per_day=1,
        weekend_focus=False,
    )

    state = AgentState(
        mode="analysis",
        hitl_mode="cli",
        days=7,
        plan_draft=plan,
        max_weekly_minutes=600,
    )

    run_case(
        "max 4 training days (should ask which days off)",
        state,
        "W tym tygodniu mogę maksymalnie 4 dni na trening.",
    )

    # Provide explicit day off + quality day + weekend time + 3 sessions/day in natural language
    run_case(
        "natural preferences extraction",
        state,
        "Chcę wolne we wtorek. Środa jakościowa. W weekend mam czas, możemy dorzucić i zrobić 3 treningi dziennie.",
    )


if __name__ == "__main__":
    main()

