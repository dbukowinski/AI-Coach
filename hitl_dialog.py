from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Tuple, Any, Optional


def _format_plan_text(plan: Dict[str, Any]) -> str:
    template = plan.get("template", "unknown")
    explanation = plan.get("explanation", "")
    sessions: List[Dict[str, Any]] = plan.get("sessions", [])

    lines: List[str] = []
    lines.append(f"Template: {template}")
    if explanation:
        lines.append("")
        lines.append("Coach summary:")
        lines.append(explanation)

    if sessions:
        lines.append("")
        lines.append("Planned sessions:")
        for session in sessions:
            date = session.get("date", "N/A")
            session_type = session.get("session_type", "N/A")
            duration = session.get("duration", "N/A")
            intensity = session.get("intensity", "N/A")
            notes = session.get("notes", "")

            lines.append("")
            lines.append(f"{date}: {session_type}")
            lines.append(f"  {duration} | {intensity}")
            if notes:
                lines.append(f"  {notes}")
    else:
        lines.append("")
        lines.append("Brak sesji w planie.")

    return "\n".join(lines)


def _format_chat(history: List[Tuple[str, str]], coach_question: str) -> str:
    lines: List[str] = []
    if history:
        for role, msg in history[-20:]:
            who = "Coach" if role == "coach" else "You"
            lines.append(f"{who}: {msg}")
            lines.append("")
    if coach_question:
        # avoid duplicating the same coach question twice (history + current)
        last = history[-1] if history else None
        if not (last and last[0] == "coach" and last[1].strip() == coach_question.strip()):
            lines.append("Coach: " + coach_question)
    return "\n".join(lines).strip()


def show_hitl_dialog(
    plan: Dict[str, Any],
    coach_question: str = "",
    history: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[bool, str]:
    """
    Pokazuje plan i pozwala użytkownikowi odpowiedzieć jak w rozmowie z trenerem.
    Zamiast klikania pól, użytkownik pisze w wolnej formie:
    - co mu pasuje,
    - czego jest za dużo / za mało,
    - które dni nie działają itd.
    """
    result: Dict[str, Any] = {
        "accepted": False,
        "feedback": "",
        "closed": False,
    }

    root = tk.Tk()
    root.title("AI Coach - Training Plan")
    root.geometry("980x780")

    def on_accept() -> None:
        result["accepted"] = True
        result["feedback"] = ""
        root.destroy()

    def on_send_feedback() -> None:
        feedback = feedback_text.get("1.0", tk.END).strip()
        result["accepted"] = False
        result["feedback"] = feedback
        root.destroy()

    def on_close() -> None:
        result["closed"] = True
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    title = ttk.Label(root, text="Here is your training plan", font=("Arial", 18, "bold"))
    title.pack(pady=(16, 10))

    plan_frame = ttk.Frame(root)
    plan_frame.pack(fill="both", expand=True, padx=14, pady=(0, 8))

    plan_text_widget = tk.Text(plan_frame, wrap="word", height=26)
    plan_scroll = ttk.Scrollbar(plan_frame, orient="vertical", command=plan_text_widget.yview)
    plan_text_widget.configure(yscrollcommand=plan_scroll.set)

    plan_text_widget.insert("1.0", _format_plan_text(plan))
    plan_text_widget.configure(state="disabled")

    plan_text_widget.pack(side="left", fill="both", expand=True)
    plan_scroll.pack(side="right", fill="y")

    # Coach dialog log
    dialog_label = ttk.Label(root, text="Coach dialog")
    dialog_label.pack(pady=(10, 6))

    dialog_frame = ttk.Frame(root)
    dialog_frame.pack(fill="both", expand=False, padx=14, pady=(0, 8))

    dialog_text = tk.Text(dialog_frame, wrap="word", height=8)
    dialog_scroll = ttk.Scrollbar(dialog_frame, orient="vertical", command=dialog_text.yview)
    dialog_text.configure(yscrollcommand=dialog_scroll.set)

    dialog_text.insert("1.0", _format_chat(history or [], coach_question))
    dialog_text.configure(state="disabled")

    dialog_text.pack(side="left", fill="both", expand=True)
    dialog_scroll.pack(side="right", fill="y")

    feedback_frame = ttk.Frame(root)
    feedback_frame.pack(fill="both", expand=False, padx=14, pady=(0, 10))

    feedback_text = tk.Text(feedback_frame, wrap="word", height=7)
    feedback_scroll = ttk.Scrollbar(feedback_frame, orient="vertical", command=feedback_text.yview)
    feedback_text.configure(yscrollcommand=feedback_scroll.set)

    feedback_text.pack(side="left", fill="both", expand=True)
    feedback_scroll.pack(side="right", fill="y")

    # Keyboard shortcuts:
    # - Enter sends the message
    # - Shift+Enter inserts a newline
    def _on_enter(event) -> str:
        on_send_feedback()
        return "break"

    def _on_shift_enter(event) -> None:
        # allow newline (default behavior)
        return None

    feedback_text.bind("<Return>", _on_enter)
    feedback_text.bind("<Shift-Return>", _on_shift_enter)

    # Start typing immediately
    feedback_text.focus_set()

    buttons = ttk.Frame(root)
    buttons.pack(pady=(6, 18))

    ttk.Button(buttons, text="Looks good, keep it", command=on_accept).grid(row=0, column=0, padx=10)
    ttk.Button(buttons, text="Send message", command=on_send_feedback).grid(row=0, column=1, padx=10)
    ttk.Button(buttons, text="Close", command=on_close).grid(row=0, column=2, padx=10)

    root.mainloop()

    if result["closed"]:
        return False, ""

    return bool(result["accepted"]), str(result["feedback"]).strip()