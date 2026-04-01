from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


ModeType = Literal["full", "analysis"]
HitlModeType = Literal["gui", "cli"]
NextActionType = Literal[
    "analyze_training",
    "generate_plan",
    "ask_user",
    "revise_plan",
    "finish",
]


@dataclass
class AgentState:
    # runtime config
    days: int = 7
    mode: ModeType = "full"

    # HITL config
    hitl_mode: HitlModeType = "gui"

    # user preferences (for more "human" plans)
    max_weekly_minutes: int = 600
    days_off: List[str] = field(default_factory=list)
    preferred_quality_days: List[str] = field(default_factory=list)
    max_sessions_per_day: int = 1  # 1, 2 lub 3 – zbilansowane RPE
    max_training_days: Optional[int] = None  # np. 4 => agent dopyta o dni wolne
    weekend_focus: bool = False  # "w weekend mam czas"
    preferences_collected: bool = False

    # conversational coach dialog (GUI/CLI)
    coach_question: str = (
        "Co chcesz zmienić w tym planie? Napisz normalnie, np. "
        "'wolne we wtorek', 'środa jakościowa', 'w weekend mam czas', "
        "'maks 4 dni treningowe', 'max 2–3 treningi dziennie'."
    )
    dialog_history: List[Tuple[str, str]] = field(default_factory=list)  # (role, text)

    # ingest / deterministic pipeline
    fetched_activity_ids: List[Any] = field(default_factory=list)
    new_activity_ids: List[Any] = field(default_factory=list)
    raw_details: List[Dict[str, Any]] = field(default_factory=list)
    clean_activities: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    # analysis artifacts
    hr_zones_summary: Dict[str, Any] = field(default_factory=dict)
    weekly_summary: Dict[str, Any] = field(default_factory=dict)
    four_week_summary: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    coach_analysis: str = ""

    # planning artifacts
    plan_draft: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    # HITL
    user_feedback: str = ""
    plan_accepted: bool = False

    # routing / loop control
    next_action: Optional[NextActionType] = None
    loop_count: int = 0
    max_loops: int = 12
    done: bool = False

    # diagnostics
    errors: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.logs.append(msg)
        print(f"[Agent] {msg}")

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.log(f"ERROR: {msg}")