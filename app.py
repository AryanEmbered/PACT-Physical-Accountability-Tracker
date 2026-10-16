#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["flask", "requests", "opencv-python", "numpy"]
# ///
from __future__ import annotations
import argparse, atexit, base64, json, logging, os, random, re, sys, threading, time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
import cv2, numpy as np, requests
from flask import Flask, Response, jsonify, request

BASE = Path(__file__).resolve().parent
SETTINGS_FILE = BASE / "settings.json"
SCHEDULE_FILE = BASE / "schedule.json"
BADGES_FILE = BASE / "badges.json"
OBSERVATIONS_FILE = BASE / "observations.json"

DEFAULT_BADGES = [
    {"id": "away", "name": "Away", "desc": "nobody is visible in the frame"},
    {"id": "at_desk", "name": "At desk", "desc": "sitting at a desk or table, working, typing or reading"},
    {"id": "sitting_idle", "name": "Sitting idle", "desc": "sitting or standing around doing nothing in particular"},
    {"id": "lying_down", "name": "Lying down", "desc": "lying down in bed or on a couch, resting or sleeping"},
    {"id": "exercising", "name": "Exercising", "desc": "doing physical exercise (workout, yoga, stretching)"},
    {"id": "meditating", "name": "Meditating", "desc": "sitting upright and very still with eyes closed"},
    {"id": "using_phone", "name": "Using phone", "desc": "looking at or holding a phone"},
    {"id": "eating", "name": "Eating", "desc": "eating or drinking"},
    {"id": "other", "name": "Other", "desc": "anything that fits none of the above"},
]
BADGE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
HM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

DEFAULT_SETTINGS = {
    "base_url": "http://127.0.0.1:1234/v1", "model": "", "api_key": "",
    "camera_index": 0, "image_width": 640, "image_height": 360, "jpeg_quality": 80,
    "window": 4, "threshold": 3, "grace_seconds": 90, "nag_cooldown": 60,
    "llm_timeout": 90, "settle_seconds": 1, "history_images": 3, "history_messages": 5, "preview": True
}
SETTING_SPECS = {
    "base_url": (str, None, None), "model": (str, None, None), "api_key": (str, None, None),
    "camera_index": (int, 0, 20), "image_width": (int, 0, 4096), "image_height": (int, 0, 2160),
    "jpeg_quality": (int, 30, 95), "window": (int, 1, 30), "threshold": (int, 1, 30),
    "grace_seconds": (int, 0, 3600), "nag_cooldown": (int, 5, 3600), "llm_timeout": (int, 5, 600),
    "settle_seconds": (int, 0, 120), "history_images": (int, 1, 8), "history_messages": (int, 0, 10),
    "preview": (bool, None, None),
}

DEFAULT_SCHEDULE = [
    {"start": "09:00", "end": "10:00", "name": "Wake up & routine", "accepts": ["at_desk", "sitting_idle", "eating"]},
    {"start": "10:00", "end": "10:30", "name": "Exercise", "accepts": ["exercising"]},
]

MESSAGES = {
    "angry": ["Hey! It's {block} time and I see {seeing}.", "{block} is scheduled. {seeing} is not {block}."],
    "suspicious": ["Are you really doing {block}? That looks like {seeing}."],
    "watching": ["Keeping an eye on {block}."], "happy": ["Nice work on {block}!"],
    "chill": ["Free time."], "paused": ["Monitoring paused."]
}

SPINNER = "|/-\\"
CONSOLE = {"url": "http://127.0.0.1:5000"}

def load_json(p, default):
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return default

def save_json(p, data):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, p)

def clean_settings(inc, cur):
    out = dict(cur)
    for k, (typ, lo, hi) in SETTING_SPECS.items():
        if k not in inc: continue
        v = inc[k]
        try:
            if typ is bool: v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")
            else:
                v = typ(v)
                if lo is not None: v = max(lo, v)
                if hi is not None: v = min(hi, v)
        except (TypeError, ValueError): continue
        out[k] = v
    out["threshold"] = min(out["threshold"], out["window"])
    return out

def clean_badges(items):
    if not isinstance(items, list): raise ValueError("Badges must be a list")
    out, seen = [], set()
    for it in items:
        if not isinstance(it, dict): continue
        name = str(it.get("name") or "").strip()[:32]
        raw_id = str(it.get("id") or "").strip().lower()
        bid = re.sub(r"[^a-z0-9]+", "_", raw_id or name.lower()).strip("_")
        if not BADGE_ID_RE.match(bid) or bid in seen: continue
        desc = str(it.get("desc") or "").strip()[:220]
        if name and desc:
            seen.add(bid)
            out.append({"id": bid, "name": name, "desc": desc})
    return out or DEFAULT_BADGES

def clean_schedule(items):
    if not isinstance(items, list): raise ValueError("Schedule must be a list")
    out = []
    for i, it in enumerate(items):
        st, en = str(it.get("start", "")), str(it.get("end", ""))
        if not (HM_RE.match(st) and HM_RE.match(en)) or st == en: continue
        acc = [a for a in it.get("accepts", []) if a in LABEL_IDS]
        if acc:
            out.append({"id": f"b{i}", "start": st, "end": en, "name": str(it.get("name") or "Block")[:60], "accepts": acc})
    return out

state_lock = threading.RLock()
settings = clean_settings(load_json(SETTINGS_FILE, {}), DEFAULT_SETTINGS)
LABELS = clean_badges(load_json(BADGES_FILE, DEFAULT_BADGES))
LABEL_IDS = [b["id"] for b in LABELS]
schedule = clean_schedule(load_json(SCHEDULE_FILE, DEFAULT_SCHEDULE))

obs_raw = load_json(OBSERVATIONS_FILE, {})
history = deque(maxlen=300)
latencies = deque(maxlen=100)
observation_history = deque((obs_raw.get("observations") or [])[-2000:], maxlen=2000)
saved_pattern = str(obs_raw.get("pattern_paragraph") or "").strip()

runtime = {
    "paused": False, "inferring": False, "infer_started": 0.0, "infer_stage": "vision",
    "error": None, "note": None, "latency": None, "changed": None, "raw": None,
    "last_label": None, "applicable": [], "description": "", "last_change": "",
    "person_visible": None, "mood": "chill", "message": "Starting…",
    "nag_id": 0, "nagging": False, "last_nag_t": 0.0, "sig": None, "analyses": 0,
    "preview": None, "preview_seq": None, "sent_seq": -1,
    "day": datetime.now().date(), "day_ok": 0, "day_total": 0,
    "last_sent_payload_summary": "None yet", "last_full_prompt": "None yet",
    "pattern_paragraph": saved_pattern or "Observing your focus and activities to construct a baseline pattern."
}

llm_state = {"model": None, "model_key": None, "frames": deque(maxlen=12), "replies": deque(maxlen=12)}

for o in list(observation_history)[-6:]:
    txt = o.get("change") or o.get("observation")
    if txt and txt not in llm_state["replies"]:
        llm_state["replies"].append(txt)

cam = None

def save_observations():
    save_json(OBSERVATIONS_FILE, {
        "observations": list(observation_history),
        "pattern_paragraph": runtime["pattern_paragraph"]
    })

def latency_stats():
    vals = list(latencies)
    if not vals: return {"count": 0, "last": None, "avg": None, "min": None, "max": None}
    return {"count": len(vals), "last": vals[-1], "avg": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}

def hm_to_min(s): return int(s[:2]) * 60 + int(s[3:])
def block_active(b, m):
    s, e = hm_to_min(b["start"]), hm_to_min(b["end"])
    return (s <= m < e) if s < e else (m >= s or m < e)

def current_block(dt):
    m = dt.hour * 60 + dt.minute
    for b in schedule:
        if block_active(b, m): return b
    return None

def block_start_dt(dt, b):
    s = hm_to_min(b["start"])
    start = dt.replace(hour=s // 60, minute=s % 60, second=0, microsecond=0)
    return start - timedelta(days=1) if start > dt else start

class Camera:
    def __init__(self, fake=False):
        self.fake = fake
        self._lock = threading.Lock()
        self._frame = None
        self._frame_ts = 0.0
        self._seq = 0
        self.error = None
        self._reopen = threading.Event()
        self._stop = threading.Event()
        threading.Thread(target=self._loop, name="camera", daemon=True).start()

    def request_reopen(self): self._reopen.set()

    def _loop(self):
        cap = None
        last_index = None
        while not self._stop.is_set():
            if self.fake:
                frame = np.random.randint(90, 140, (480, 640, 3), dtype=np.uint8)
                x = int(time.time() * 90) % 480
                cv2.rectangle(frame, (x, 190), (x + 120, 400), (60, 110, 210), -1)
                time.sleep(0.06)
            else:
                idx = int(settings["camera_index"])
                if cap is None or self._reopen.is_set() or idx != last_index:
                    self._reopen.clear()
                    if cap: cap.release()
                    last_index = idx
                    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
                    cap = cv2.VideoCapture(idx, backend)
                    if not cap.isOpened():
                        self.error = f"Cannot open camera {idx}"
                        time.sleep(1.5)
                        cap = None
                        continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    if cap: cap.release()
                    cap = None
                    self.error = "Camera read failed"
                    time.sleep(1.0)
                    continue
                self.error = None
            with self._lock:
                self._frame = frame
                self._frame_ts = time.time()
                self._seq += 1

    def latest(self):
        with self._lock:
            if self._frame is None or time.time() - self._frame_ts > 8: return None
            return self._frame.copy(), self._seq, self._frame_ts

def encode_frame(frame, width, height, quality):
    h, w = frame.shape[:2]
    scale = min([width / w if width else 1.0, height / h if height else 1.0])
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok: raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

def auth_headers(s):
    h = {"Content-Type": "application/json"}
    if s["api_key"]: h["Authorization"] = "Bearer " + s["api_key"]
    return h

def resolve_model(s):
    if s["model"]: return s["model"]
    key = s["base_url"]
    if llm_state["model"] and llm_state["model_key"] == key: return llm_state["model"]
    r = requests.get(s["base_url"].rstrip("/") + "/models", headers=auth_headers(s), timeout=10)
    r.raise_for_status()
    models = [m["id"] for m in r.json().get("data", [])]
    if not models: raise RuntimeError("No models available on server.")
    llm_state["model"], llm_state["model_key"] = models[0], key
    return models[0]

def build_prompt(total_frames=1, prev_replies=None):
    lines = "\n".join(f"- {b['id']}: {b['desc']}" for b in LABELS)
    prev_replies = [r for r in (prev_replies or []) if r and r.strip()]
    if prev_replies:
        numbered = "\n".join(f'{i+1}. "{r.strip()}"' for i, r in enumerate(prev_replies))
        anti_repeat_block = (
            "CRITICAL ANTI-REPETITION CONSTRAINT:\n"
            "You are FORBIDDEN from outputting any of the following previous sentences or phrasing them nearly identically:\n"
            f"{numbered}\n"
            "If the person is still doing the exact same thing or has not moved, DO NOT repeat the previous sentence. "
            "Instead, describe a different detail (posture, lighting, expression) or explicitly state that there wasn't any difference.\n\n"
        )
    else:
        anti_repeat_block = ""

    if total_frames > 1:
        task_instruction = (
            f"You are given {total_frames} images arranged in strict CHRONOLOGICAL order from oldest to newest.\n"
            "Image 1 is the oldest past frame. The LAST image is the CURRENT (latest) frame.\n"
            "Analyze and answer ONLY for the CURRENT (last) frame by comparing it against the earlier frames:\n"
            "1. Pay strict attention to physical additions/removals or movements\n"
            "2. Prioritize the differences between frames.\n"
            "3. DO NOT hallucinate features from past images onto the current image."
        )
    else:
        task_instruction = (
            "You are given a single CURRENT webcam image. There are no earlier frames.\n"
            "Directly describe what the person is doing, their exact visible posture, and accessories currently worn."
        )

    return (
        f"/no_think\nYou are an ultra-precise focus & routine vision monitor. Output ONLY JSON.\n\n"
        f"{task_instruction}\n\n"
        f"Allowed Activity Badges:\n{lines}\n\n"
        f"{anti_repeat_block}"
        "Schema:\n"
        "{\n"
        '  "applicable": ["badge_id"],\n'
        '  "activity": "badge_id",\n'
        '  "person_visible": boolean,\n'
        '  "changed": boolean,\n'
        '  "observation": "one sentence describing the CURRENT frame accurately without repeating forbidden sentences",\n'
        '  "change": "one sentence describing physical difference from earlier frames, or empty string if no change"\n'
        "}"
    )

def parse_observation(text):
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    if start < 0: raise RuntimeError(f"Model reply was not JSON: {text[:120]}")
    result, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(result, dict) or result.get("activity") not in LABEL_IDS:
        raise RuntimeError(f"Invalid activity in model reply: {text[:120]}")
    app = [a for a in result.get("applicable", []) if a in LABEL_IDS]
    if result["activity"] not in app: app.insert(0, result["activity"])
    return (result["activity"], app, str(result.get("observation", "")).strip()[:500],
            str(result.get("change", "")).strip()[:500], bool(result.get("person_visible", False)),
            bool(result.get("changed", False)))

def classify(jpeg, frame_seq, s, prev_items=None, prev_replies=None):
    prev_items = list(prev_items or [])
    total_imgs = len(prev_items) + 1
    model = resolve_model(s)
    prompt_text = build_prompt(total_frames=total_imgs, prev_replies=prev_replies)

    parts = [{"type": "text", "text": prompt_text}]
    summary_parts = [f"Text prompt ({len(prompt_text)} chars)"]

    for i, (p_seq, p_bytes) in enumerate(prev_items):
        seq_idx = i + 1
        ago = total_imgs - seq_idx
        tag = f"[IMAGE {seq_idx} of {total_imgs} - CHRONOLOGICAL PAST FRAME (#{p_seq}, {ago} step(s) ago)]:"
        parts.append({"type": "text", "text": tag})
        parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(p_bytes).decode()}})
        summary_parts.append(f"Past Image #{p_seq} ({len(p_bytes)}B)")

    tag_cur = f"[IMAGE {total_imgs} of {total_imgs} - CURRENT LATEST FRAME (#{frame_seq}, answer for this)]:"
    parts.append({"type": "text", "text": tag_cur})
    parts.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})
    summary_parts.append(f"Current Image #{frame_seq} ({len(jpeg)}B)")

    with state_lock:
        runtime["last_full_prompt"] = prompt_text
        runtime["last_sent_payload_summary"] = " -> ".join(summary_parts)

    payload = {
        "model": model, "messages": [{"role": "user", "content": parts}],
        "temperature": 0.7, "presence_penalty": 0.6, "frequency_penalty": 0.5, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "observation", "strict": True,
            "schema": {"type": "object", "additionalProperties": False,
                       "required": ["applicable", "activity", "person_visible", "changed", "observation", "change"],
                       "properties": {"applicable": {"type": "array", "items": {"type": "string", "enum": LABEL_IDS}},
                                      "activity": {"type": "string", "enum": LABEL_IDS},
                                      "person_visible": {"type": "boolean"}, "changed": {"type": "boolean"},
                                      "observation": {"type": "string"}, "change": {"type": "string"}}}}}
    }
    r = requests.post(s["base_url"].rstrip("/") + "/chat/completions", json=payload, headers=auth_headers(s), timeout=s["llm_timeout"])
    if not r.ok: raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    msg = r.json()["choices"][0].get("message") or {}
    content = msg.get("content") or ""
    if isinstance(content, list): content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not content: raise RuntimeError("Model returned no content.")

    label, applicable, desc, chg, p_vis, changed = parse_observation(content)
    if not p_vis and "away" in LABEL_IDS:
        label, applicable = "away", ["away"]

    with state_lock:
        runtime["raw"] = content[:200]
        llm_state["frames"].append((frame_seq, jpeg))
        stored_text = desc.strip()
        if stored_text and (not llm_state["replies"] or llm_state["replies"][-1] != stored_text):
            llm_state["replies"].append(stored_text)

    return label, applicable, desc, chg, p_vis, changed

def update_long_term_pattern(s):
    """Text-only synthesis: updates the single evolving paragraph summarizing routine & habits."""
    recent_5 = list(observation_history)[-5:]
    if len(recent_5) < 3: return
    model = resolve_model(s)
    current_para = runtime["pattern_paragraph"]

    obs_lines = []
    for o in recent_5:
        tm = datetime.fromtimestamp(o["t"]).strftime("%H:%M:%S")
        obs_lines.append(f"[{tm}] Activity: {o['label']}. Detail: {o.get('observation','')}. Physical Change: {o.get('change','None')}")
    obs_text = "\n".join(obs_lines)

    prompt = (
        "You are an observant behavioral routine biographer.\n"
        "Your goal is to maintain a single ongoing, cohesive paragraph (3 to 4 sentences maximum) "
        "describing the user's ongoing physical habits, workspace posture, focus flow, and transitions.\n\n"
        f"EXISTING PATTERN PARAGRAPH:\n\"{current_para}\"\n\n"
        f"NEW OBSERVATIONS (LAST 5 FRAMES):\n{obs_text}\n\n"
        "TASK:\n"
        "Synthesize the new observations into the existing narrative. Rewrite it into ONE single cohesive paragraph of natural English prose. Do not ever say the words, The User. whatever you are saying will be shown to the user, so don't say the user wants me to.\n"
        "RULES:\n"
        "- Output ONLY the final revised paragraph of pure plain text.\n"
        "- DO NOT output JSON. DO NOT output braces {}, brackets, lists, bullets, or labels.\n"
        "- Keep it grounded and natural."
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    try:
        r = requests.post(s["base_url"].rstrip("/") + "/chat/completions", json=payload, headers=auth_headers(s), timeout=35)
        if r.ok:
            msg = r.json()["choices"][0].get("message") or {}
            raw = str(msg.get("content") or "").strip()
            # Clean accidental markdown or JSON wrappers
            raw = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
            # If the model tried to output a JSON object, unpack it into plain sentences
            if raw.startswith("{") and raw.endswith("}"):
                try:
                    parsed = json.loads(raw)
                    raw = " ".join(str(v) for v in parsed.values() if v and isinstance(v, str))
                except Exception:
                    raw = ""
            raw = re.sub(r"\s+", " ", raw).strip()
            if len(raw) > 20 and not raw.startswith("{"):
                with state_lock:
                    runtime["pattern_paragraph"] = raw
                save_observations()
    except Exception as ex:
        runtime["note"] = f"Pattern update skipped: {ex}"

def record(label, applicable, now, observation, change, changed, frame_seq=None):
    b = current_block(now)
    ok = (label in b["accepts"]) if b else None
    if now.date() != runtime["day"]: runtime.update(day=now.date(), day_ok=0, day_total=0)
    if ok is not None:
        runtime["day_total"] += 1
        runtime["day_ok"] += 1 if ok else 0
    entry = {"t": now.timestamp(), "label": label, "applicable": list(applicable), "ok": ok,
             "changed": bool(changed or change), "block": b["id"] if b else None, "observation": observation,
             "change": change, "seq": frame_seq}
    history.append(entry)
    observation_history.append(entry)
    runtime.update(last_label=label, description=observation, last_change=change, changed=bool(changed or change))
    runtime["analyses"] += 1
    if changed or change or runtime["analyses"] % 10 == 0:
        save_observations()

def evaluate():
    with state_lock:
        now = datetime.now()
        s = settings
        b = current_block(now)
        lbl = runtime["last_label"]
        nagged = False
        if runtime["paused"] or (cam and cam.error): mood, violating = "paused", False
        elif b is None: mood, violating = "chill", False
        else:
            start = block_start_dt(now, b)
            in_blk = [e for e in history if e["block"] == b["id"] and e["t"] >= start.timestamp()]
            recent = in_blk[-s["window"]:]
            bad = sum(1 for e in recent if not e["ok"])
            violating = (bool(recent) and (now - start).total_seconds() >= s["grace_seconds"]
                         and len(recent) >= s["threshold"] and bad >= s["threshold"] and not recent[-1]["ok"])
            if violating: mood = "angry"
            elif not in_blk: mood = "watching"
            else:
                last8 = in_blk[-8:]
                r = sum(1 for e in last8 if e["ok"]) / len(last8)
                mood = "happy" if r >= 0.75 else ("watching" if r >= 0.5 else "suspicious")

        if violating and time.time() - runtime["last_nag_t"] >= s["nag_cooldown"]:
            runtime["nag_id"] += 1
            runtime["last_nag_t"] = time.time()
            nagged = True
        runtime["nagging"] = violating
        sig = (mood, b["id"] if b else None, lbl)
        if nagged or sig != runtime["sig"]:
            runtime["sig"] = sig
            runtime["mood"] = mood
            runtime["message"] = random.choice(MESSAGES[mood]).format(block=b["name"] if b else "routine", seeing=(lbl or "nothing").replace("_", " "))

def analyzer():
    while True:
        if runtime["paused"] or (cam and cam.error):
            runtime["inferring"] = False
            time.sleep(0.4)
            continue
        snap = cam.latest()
        if snap is None:
            runtime["inferring"] = False
            time.sleep(0.3)
            continue
        frame, frame_seq, frame_ts = snap
        if frame_seq == runtime["sent_seq"]:
            runtime["inferring"] = False
            time.sleep(0.05)
            continue

        with state_lock: s = dict(settings)
        runtime["inferring"] = True
        runtime["infer_stage"] = "vision"
        runtime["infer_started"] = time.time()
        t0 = time.time()
        label = desc = chg = p_vis = changed = err = None
        applicable = []

        try:
            jpeg = encode_frame(frame, s["image_width"], s["image_height"], s["jpeg_quality"])
            with state_lock:
                runtime["sent_seq"] = frame_seq
                n_img = max(1, int(s["history_images"]))
                n_msg = max(0, int(s["history_messages"]))
                prev_items = list(llm_state["frames"])[-(n_img - 1):] if n_img > 1 else []
                prev_replies = list(llm_state["replies"])[-n_msg:] if n_msg else []
            label, applicable, desc, chg, p_vis, changed = classify(jpeg, frame_seq, s, prev_items, prev_replies)
        except Exception as ex:
            err = str(ex)[:400]

        latency = time.time() - t0
        with state_lock:
            runtime["latency"] = latency
            runtime["error"] = err
            if label:
                runtime["preview"] = jpeg
                runtime["preview_seq"] = frame_seq
                latencies.append(latency)
                runtime["person_visible"] = p_vis
                runtime["applicable"] = applicable
                record(label, applicable, datetime.fromtimestamp(frame_ts), desc, chg, changed, frame_seq)
        evaluate()

        # Update pattern paragraph text-only every 5 inferences
        if runtime["analyses"] > 0 and runtime["analyses"] % 5 == 0 and not err:
            with state_lock:
                runtime["infer_stage"] = "pattern"
            update_long_term_pattern(s)

        with state_lock:
            runtime["inferring"] = False

def ticker():
    while True:
        time.sleep(1)
        evaluate()

# ----------------------------------------------------------------------------
# Console Dashboard
# ----------------------------------------------------------------------------
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m|\x1b\]8;;[^\x1b]*(?:\x1b\\|\x07)")
STATE_COLOR = {"on track": "32", "watching": "36", "slipping": "33", "OFF TASK": "1;31",
               "free time": "2", "paused": "2", "camera offline": "31", "model error": "31"}

def _vis(s): return ANSI_RE.sub("", s)
def _c(code, text): return f"\x1b[{code}m{text}\x1b[0m" if code else text
def _wrap(text, width):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if not cur: cur = w
        elif len(cur) + 1 + len(w) <= width: cur += " " + w
        else: lines.append(cur); cur = w
    lines.append(cur)
    return lines or [""]

def render_console(tick, inner=72):
    label_w, prefix = 13, 16
    with state_lock:
        now = datetime.now()
        b = current_block(now)
        inf = runtime["inferring"]
        stage = runtime["infer_stage"]
        elap = (time.time() - runtime["infer_started"]) if inf else 0.0
        seq = runtime["preview_seq"]
        last_sent = runtime["last_sent_payload_summary"]
        st_state = "paused" if runtime["paused"] else ("camera offline" if (cam and cam.error) else ("model error" if runtime["error"] else ({"angry": "OFF TASK", "suspicious": "slipping", "watching": "watching", "happy": "on track", "chill": "free time", "paused": "paused"}.get(runtime["mood"], "..."))))

        st = {
            "clock": now.strftime("%H:%M:%S"), "block": b, "state": st_state,
            "inferring": inf, "stage": stage, "elapsed": elap, "inference_id": runtime["analyses"],
            "frame_seq": seq, "activity": (runtime["last_label"] or "-").replace("_", " "),
            "applies": ", ".join(a.replace("_", " ") for a in runtime["applicable"]) or "-",
            "person": (("visible" if runtime["person_visible"] else "not visible") if runtime["person_visible"] is not None else "-"),
            "changed": ("yes" if runtime["changed"] else "no") if runtime["changed"] is not None else "-",
            "latency": f"{runtime['latency']:.1f}s" if runtime["latency"] else "-",
            "observed": runtime["description"] or "",
            "change": runtime["last_change"] or "",
            "pattern": runtime["pattern_paragraph"],
            "error": runtime["error"],
            "payload": last_sent,
        }

    def field(label, value, code=None):
        pieces = _wrap(value, inner - prefix)
        out = []
        for i, p in enumerate(pieces):
            pre = (" " + label.ljust(label_w) + "  ") if i == 0 else " " * prefix
            out.append(pre + _c(code, p))
        return out

    title, clock = "PACT", st["clock"]
    fill = max(1, inner - 6 - len(title) - len(clock))
    top = "\u250c\u2500 " + _c("1;36", title) + " " + "\u2500" * fill + " " + _c("1", clock) + " \u2500\u2510"
    mid = "\u251c" + "\u2500" * inner + "\u2524"
    bottom = "\u2514" + "\u2500" * inner + "\u2518"

    body = [""]
    body += field("routine", f"{st['block']['name']} ({st['block']['start']}\u2013{st['block']['end']})" if st['block'] else "free time \u2014 nothing scheduled")
    body += field("state", st["state"], STATE_COLOR.get(st["state"]))

    # Isolated live status line
    if st["inferring"]:
        sub_desc = "vision" if st["stage"] == "vision" else "updating pattern synopsis"
        body.append(" " + _c("33", SPINNER[tick % len(SPINNER)]) + " " + _c("33", f"running {sub_desc}\u2026") + f" ({st['elapsed']:.1f}s)")
    else:
        body.append(" " + _c("32", "\u2022") + " " + _c("2", "ready / waiting"))

    body_lines = []
    for s in body:
        body_lines.append("\u2502" + s + " " * max(0, inner - len(_vis(s))) + "\u2502")

    # Lower section: preserved last inference
    lower = [
        " " + _c("1", f"Latest Inference #{st['inference_id']}") + _c("2", f" (frame #{st['frame_seq'] if st.get('frame_seq') is not None else '-'})")
    ]
    lower += field("activity", st["activity"])
    lower += field("applies", st["applies"])
    lower += field("person", st["person"])
    lower += field("changed", st["changed"], "33" if st["changed"] == "yes" else None)
    lower += field("latency", st["latency"])
    lower += field("observed", st["observed"] or "\u2014")
    lower += field("change", st["change"] or "\u2014")
    lower += field("pattern", st["pattern"], "35")
    lower.append("")
    lower += field("payload", st["payload"], "34")
    if st["error"]:
        lower += field("error", st["error"], "31")
    lower += field("web ui", CONSOLE["url"])
    lower.append("")

    lower_lines = ["\u2502" + s + " " * max(0, inner - len(_vis(s))) + "\u2502" for s in lower]

    return "\n".join([top] + body_lines + [mid] + lower_lines + [bottom])

def console_loop():
    tick = 0
    prev = None
    while True:
        tick += 1
        text = render_console(tick)
        if text != prev:
            prev = text
            sys.stdout.write("\x1b[2J\x1b[3J\x1b[H" + text + "\n")
            sys.stdout.flush()
        time.sleep(0.2)

# ----------------------------------------------------------------------------
# Flask App
# ----------------------------------------------------------------------------
app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

@app.get("/")
def index():
    return Response((BASE / "index.html").read_text(encoding="utf-8"), mimetype="text/html", headers={"Cache-Control": "no-store"})

@app.get("/api/state")
def api_state():
    with state_lock:
        now = datetime.now()
        b = current_block(now)
        total = runtime["day_total"]
        hist = [{"time": datetime.fromtimestamp(e["t"]).strftime("%H:%M:%S"), "label": e["label"],
                 "ok": e["ok"], "changed": e.get("changed"), "observation": e.get("observation", ""),
                 "change": e.get("change", ""), "seq": e.get("seq")} for e in list(history)[-40:]]
        recent = [{"time": datetime.fromtimestamp(e["t"]).strftime("%H:%M:%S"), "activity": e.get("label"),
                   "observation": e.get("observation", ""), "change": e.get("change", ""),
                   "ok": e.get("ok"), "seq": e.get("seq")} for e in list(observation_history)[-5:][::-1]]
        return jsonify({
            "now": now.strftime("%H:%M:%S"), "block": b, "schedule": schedule, "settings": settings,
            "badges": LABELS, "paused": runtime["paused"], "camera_error": (cam.error if cam else "no camera"),
            "error": runtime["error"], "note": runtime["note"], "latency": runtime["latency"],
            "latency_stats": latency_stats(), "changed": runtime["changed"], "inferring": runtime["inferring"],
            "infer_stage": runtime["infer_stage"], "inference_id": runtime["analyses"],
            "activity": runtime["last_label"], "applicable": runtime["applicable"],
            "person_visible": runtime["person_visible"], "description": runtime["description"], "change": runtime["last_change"],
            "pattern_paragraph": runtime["pattern_paragraph"],
            "mood": runtime["mood"], "message": runtime["message"], "nag_id": runtime["nag_id"], "nagging": runtime["nagging"],
            "day_score": round(100 * runtime["day_ok"] / total) if total else None, "day_total": total,
            "history": hist, "recent": recent,
            "has_preview": runtime["preview"] is not None, "frame_seq": runtime["preview_seq"],
            "last_prompt": runtime["last_full_prompt"]
        })

@app.put("/api/schedule")
def api_schedule():
    global schedule
    try: schedule = clean_schedule((request.get_json(silent=True) or {}).get("schedule"))
    except ValueError as e: return jsonify({"error": str(e)}), 400
    with state_lock:
        history.clear(); runtime["sig"] = None
        save_json(SCHEDULE_FILE, schedule)
    evaluate()
    return jsonify({"ok": True, "schedule": schedule})

@app.put("/api/badges")
def api_badges():
    global LABELS, LABEL_IDS
    try: new = clean_badges((request.get_json(silent=True) or {}).get("badges"))
    except ValueError as e: return jsonify({"error": str(e)}), 400
    with state_lock:
        LABELS, LABEL_IDS = new, [b["id"] for b in new]
        save_json(BADGES_FILE, LABELS)
    return jsonify({"ok": True, "badges": LABELS})

@app.put("/api/settings")
def api_settings():
    global settings
    with state_lock:
        old_cam = settings["camera_index"]
        settings = clean_settings(request.get_json(silent=True) or {}, settings)
        llm_state.update(model=None, model_key=None)
        latencies.clear()
        save_json(SETTINGS_FILE, settings)
        if settings["camera_index"] != old_cam and cam: cam.request_reopen()
    return jsonify({"ok": True, "settings": settings})

@app.post("/api/clear")
def api_clear():
    with state_lock:
        history.clear(); observation_history.clear()
        runtime["applicable"] = []; runtime["sig"] = None
        runtime["pattern_paragraph"] = "Observing your focus and activities to construct a baseline pattern."
        save_observations()
    return jsonify({"ok": True})

@app.post("/api/control")
def api_control():
    act = (request.get_json(silent=True) or {}).get("action")
    with state_lock:
        if act in ("pause", "resume"):
            runtime["paused"] = act == "pause"
            runtime["sig"] = None
    evaluate()
    return jsonify({"ok": True, "paused": runtime["paused"]})

@app.get("/api/frame.jpg")
def api_frame():
    with state_lock: jpeg = runtime["preview"] if settings["preview"] else None
    if jpeg is None: return Response(status=204)
    return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

def main():
    global cam
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--fake-camera", action="store_true")
    args = ap.parse_args()

    if os.name == "nt": os.system("")
    atexit.register(save_observations)
    CONSOLE["url"] = f"http://{args.host}:{args.port}"

    cam = Camera(fake=args.fake_camera)
    threading.Thread(target=analyzer, daemon=True, name="analyzer").start()
    threading.Thread(target=ticker, daemon=True, name="ticker").start()
    threading.Thread(target=console_loop, daemon=True, name="console").start()
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()