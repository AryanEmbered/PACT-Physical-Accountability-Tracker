# PACT — Physical Accountability Continuous Tracker

Local-first webcam focus & routine monitor. Uses a vision LLM (OpenAI-compatible API) to watch your webcam, classify what you're doing against customizable activity badges, compare it to your daily schedule, and nudge you when you're off-task — via a cozy cat avatar web UI + console dashboard.

## How it works

1. Captures webcam frames with OpenCV.
2. Sends current frame + recent history to a vision model (`base_url`, e.g. LM Studio at `http://127.0.0.1:1234/v1`).
3. Model returns JSON: `activity`, `applicable` badges, `observation`, `change`, `person_visible`, `changed`.
4. `app.py` compares activity to the active schedule block (`window` / `threshold` / `grace_seconds` logic), updates mood (`happy / watching / suspicious / angry / chill / paused`), and shows it in:
   - Web UI at `http://127.0.0.1:5000` (`index.html` + `/api/*`)
   - Terminal dashboard (PACT box with live spinner, latency, pattern paragraph)
5. Every 5 inferences it rewrites a single `pattern_paragraph` summarizing your routine/habits (stored in `observations.json`).

Default badges: `away, at_desk, sitting_idle, lying_down, exercising, meditating, using_phone, eating, other` — all editable in the UI.

## Requirements

- Windows 10/11 (Linux/macOS works with `python app.py` too), webcam
- An OpenAI-compatible vision server running locally, e.g.:
  - LM Studio → start server at `http://127.0.0.1:1234/v1` with a vision model loaded
  - Or any ` /v1/chat/completions` + `/v1/models` endpoint
- Option A (easiest): `uv` — no manual Python install needed, handled by `run.bat`
- Option B: Python >= 3.10 + `pip install -r requirements.txt`

## Quick start

### Option A — double-click (Windows)

1. Download/clone this repo.
2. Double-click `run.bat`.
   - Uses the bundled `uv.exe` if present, otherwise system `uv` from PATH.
   - Runs `uv run app.py` — auto-creates env and installs `flask, requests, opencv-python, numpy` from the PEP 723 header in `app.py`.
3. Open `http://127.0.0.1:5000`.

### Option B — manual

```bat
pip install -r requirements.txt
python app.py
rem flags: --host 127.0.0.1 --port 5000 --fake-camera
```

Then open `http://127.0.0.1:5000` → Edit → set Model server URL, camera index, schedule blocks, badges.

## Configuration

All editable in Web UI → Edit (persisted as JSON next to `app.py`):

- `settings.json` — `base_url, model, api_key, camera_index, image_width/height, jpeg_quality, window, threshold, grace_seconds, nag_cooldown, llm_timeout, history_images, history_messages, preview`
- `schedule.json` (auto-created) — `[{start:"09:00", end:"10:00", name:"...", accepts:["at_desk", ...]}]`
- `badges.json` (auto-created) — `[{id:"at_desk", name:"At desk", desc:"..."}]`
- `observations.json` (auto-created, local only) — inference log + `pattern_paragraph`

## Files in this repo

| File | Purpose | Upload to GitHub? |
|------|---------|-------------------|
| `app.py` | Backend: camera, LLM calls, Flask API, console UI | Yes |
| `index.html` | Frontend web UI | Yes |
| `run.bat` | Windows launcher (`uv run app.py`) | Yes |
| `requirements.txt` | `flask, requests, opencv-python, numpy` for pip users | Yes |
| `settings.json` | Default settings (no secrets) | Yes |
| `README.md` | This file | Yes |
| `.gitignore` | Excludes local/cache files | Yes |

Do **not** commit: `uv.exe / uvw.exe / uvx.exe`, `.uv/`, `__pycache__/`, `observations.json`, `schedule.json`, `badges.json` (personal).

## API (Flask)

- `GET /` → UI, `GET /api/state`, `GET /api/frame.jpg`
- `PUT /api/schedule`, `PUT /api/badges`, `PUT /api/settings`
- `POST /api/clear`, `POST /api/control {action: pause|resume}`

## Privacy

All video stays local except frames sent to *your* configured `base_url`. `observations.json` is your personal log — keep it git-ignored.

## License

MIT — do what you want, no warranty. Your webcam data is yours.
