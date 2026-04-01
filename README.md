# AI Coach – Strava Training Plan Agent (LangGraph + HITL)

An interactive “AI Coach” that syncs training data from **Strava**, runs a lightweight training analysis, and generates a personalized weekly plan.  
The agent is orchestrated as a **LangGraph** workflow with a **Human-in-the-Loop** dialog (GUI or CLI).

## Features

- **Strava sync**: OAuth tokens (refresh flow), fetch last N days, dedupe by activity id, persist raw data locally.
- **Preprocessing**: normalize raw activities → clean dataset.
- **Training analysis**: weekly + 4-week load, HR-zone based intensity buckets, simple flags (risk signals).
- **Plan generation**: recovery / balanced / build templates with personalization:
  - days off
  - preferred “quality day”
  - weekend focus (more time available)
  - max weekly time budget
  - 1–3 sessions per day (balanced RPE)
- **Interactive coaching loop (HITL)**:
  - GUI (Tkinter) dialog like a coach
  - CLI mode for terminal-only usage (`--no-gui`)
- **Optional LLM**: Amazon Bedrock for coaching-style summaries and plan revision (fallback to deterministic rules if AWS is not available).
- **Export**: saves the final plan to a **CSV** (Excel/Google Sheets friendly) in `data/`.

## Project structure (agent-related)

- `run_agent.py`: entry point (CLI)
- `agent_graph.py`: LangGraph workflow definition
- `agent_nodes.py`: pipeline + reasoning nodes and coach dialog logic
- `agent_state.py`: state model
- `hitl_dialog.py`: Tkinter dialog (HITL)
- `sync_strava.py`: Strava integration + local persistence
- `clean.py`: raw → clean normalization
- `report.py`: simple summary report (JSON/MD)
- `analysis.py`: training analysis (weekly + 4-week aggregates, flags)
- `plan_generator.py`: plan templates + personalization logic

> Note: the repo also contains workshop artifacts under `Amazon Bedrock old/`.

## Requirements

- Python 3.10+ recommended (works on Windows/macOS/Linux)
- (Optional) AWS CLI configured for Bedrock usage

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash on Windows
pip install -r requirements.txt
```

## Strava credentials

Create a `.env` file in the repository root (do **not** commit it):

```env
STRAVA_CLIENT_ID=...
STRAVA_CLIENT_SECRET=...
STRAVA_ACCESS_TOKEN=...
STRAVA_REFRESH_TOKEN=...
STRAVA_EXPIRES_AT=...
```

The agent uses `python-dotenv` to load `.env` automatically, and it will update tokens in-place when refreshing.

## Run

### Full pipeline (sync → clean → report → analysis → plan → dialog)

GUI (Tkinter):

```bash
python run_agent.py --mode full --days 7
```

CLI (terminal-only):

```bash
python run_agent.py --mode full --days 7 --no-gui
```

### Analysis-only mode

Skips Strava sync/clean/report and goes directly into the reasoning loop (useful if you already have data in `data/`):

```bash
python run_agent.py --mode analysis --days 7
```

## Optional: Amazon Bedrock

If you want the agent to use Bedrock for coaching text + plan revision, export AWS environment variables first.
This repo includes `aws_login.sh` (SSO profile example).

```bash
source aws_login.sh
python run_agent.py --mode full --days 7
```

If AWS credentials are missing/expired, the agent falls back to deterministic logic.

## Outputs

Generated artifacts (ignored by git) are written to `data/`, e.g.:

- `activities_raw.json`
- `activities_clean.json`
- `weekly_summary.json`
- `four_week_summary.json`
- `flags.json`
- `training_plan_current.json`
- `training_plan_history.json`
- `training_plan_YYYYMMDD_HHMMSS.csv` (Excel export)

## Testing (quick)

A small helper script exists to exercise the dialog flow without Strava:

```bash
python test_dialog_flow.py
```

## Notes on safety and training load

This is a personal project / prototype. It does not replace medical or professional coaching advice.

