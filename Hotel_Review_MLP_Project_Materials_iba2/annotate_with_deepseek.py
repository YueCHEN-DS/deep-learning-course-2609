"""LLM-assisted labeling with DeepSeek.

Uses the DeepSeek chat API to propose 8 aspect × 5 state labels for hotel
sentences. Outputs are AI suggestions only — a human must review them before
they become training data.

Usage:
    export DEEPSEEK_API_KEY=sk-...
    # or leave unset and the script reads api_kyes.txt next to this file

    python annotate_with_deepseek.py --limit 20
    python annotate_with_deepseek.py --start 0 --limit 50 --out labels_ai.jsonl
    python annotate_with_deepseek.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Install openai first:  python -m pip install -U openai", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parent
SENTENCES_CSV = ROOT / "hotel_sentences_2000.csv"
RECORDS_TO_REVIEW = ROOT / "records_to_review.csv"
API_KEYS_FILE = ROOT / "api_keys.txt" if (ROOT / "api_keys.txt").exists() else ROOT / "api_kyes.txt"

# Official DeepSeek. Override with DEEPSEEK_BASE_URL if your course uses a proxy.
DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")

ASPECTS = [
    "cleanliness",
    "service",
    "location",
    "facilities",
    "room_comfort",
    "sound_insulation_noise",
    "food",
    "value",
]
STATES = {
    0: "absent",
    1: "negative",
    2: "neutral",
    3: "positive",
    4: "mixed",
}

PROMPT_SYSTEM = """You are an expert annotator for a Chinese hotel-review aspect-sentiment project.

This is a simple classification task. Do not write a long chain of thought.
Decide quickly from the rules below and return only the JSON object.

Label ONE sentence. For EACH of the 8 aspects, assign exactly one state code.

Aspects (fixed order, index 0-7):
0 cleanliness — dirt, hygiene, stains, pests
1 service — staff behavior, help, check-in / check-out handling
2 location — access, transport, nearby places
3 facilities — equipment, amenities, condition (AC, elevator, bathroom fixtures, etc.)
4 room_comfort — space, bed comfort, temperature comfort
5 sound_insulation_noise — soundproofing, hallway / traffic / equipment noise, quietness
6 food — hotel breakfast and dining quality (taste, freshness, variety of hotel food)
7 value — prices, charges, value for money

States:
0 absent — aspect not mentioned
1 negative — unfavorable evaluation
2 neutral — mentioned, no clear positive/negative view (e.g. "早餐一般" → food=2)
3 positive — favorable evaluation
4 mixed — both positive and negative views of THIS same aspect in one sentence

Boundary rules:
- Label the property discussed, not every object word.
- 空调坏了 → facilities=1 (equipment broken)
- 空调太吵 → sound_insulation_noise=1 (noise)
- 床单很脏 → cleanliness=1
- 酒店早餐很贵 → value=1
- 酒店早餐很难吃 → food=1
- 附近有很多餐厅 → location=3
- "前台热情，但办理太慢" → service=4 (mixed, same aspect)
- "位置很好，但服务很差" → location=3, service=1 (different aspects)
- "以后不会再住了" with no specific aspect → all absent
- Overall rating words alone do not invent missing aspects
- If text is a hotel reply, unusable fragment, or too ambiguous → set exclude=true and still fill best-effort states (usually all 0)

Return ONLY a single JSON object, no markdown fences, matching:
{
  "sentence_id": "...",
  "states": [s0, s1, s2, s3, s4, s5, s6, s7],
  "evidence": {
    "cleanliness": "exact span or empty",
    "service": "...",
    "location": "...",
    "facilities": "...",
    "room_comfort": "...",
    "sound_insulation_noise": "...",
    "food": "...",
    "value": "..."
  },
  "exclude": false,
  "needs_review": false,
  "notes": "short reason if exclude/needs_review or mixed"
}

Evidence must be an exact substring of the input sentence (or "").
states[i] must be an integer 0-4.
"""

USER_TEMPLATE = """sentence_id: {sid}
sentence: {sentence}

context_before: {before}
context_after: {after}

Return the JSON annotation now."""


def load_api_key() -> str:
    env = os.getenv("DEEPSEEK_API_KEY")
    if env:
        return env.strip()
    if API_KEYS_FILE.exists():
        text = API_KEYS_FILE.read_text(encoding="utf-8")
        # Prefer the line labeled deepseek
        for line in text.splitlines():
            if "deepseek" in line.lower() and "sk-" in line:
                m = re.search(r"(sk-[A-Za-z0-9]+)", line)
                if m:
                    return m.group(1)
        m = re.search(r"(sk-[A-Za-z0-9]+)", text)
        if m:
            # first key in file is embedding key; skip if a deepseek-looking second exists
            keys = re.findall(r"(sk-[A-Za-z0-9]+)", text)
            if len(keys) >= 2:
                return keys[1]
            return keys[0]
    raise SystemExit(
        "No API key. Set DEEPSEEK_API_KEY or put the key in api_kyes.txt"
    )


def load_sentences() -> list[dict]:
    with SENTENCES_CSV.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_exclude_ids() -> set[str]:
    """Optional: if records_to_review.csv has a decision column, honor exclude."""
    if not RECORDS_TO_REVIEW.exists():
        return set()
    with RECORDS_TO_REVIEW.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    exclude = set()
    for r in rows:
        decision = (r.get("decision") or r.get("record_decision") or "").strip().lower()
        sid = r.get("sentence_id") or ""
        if sid and decision == "exclude":
            exclude.add(sid)
    return exclude


def parse_json_reply(text: str, sentence_id: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object in reply: {text[:200]!r}")
    obj = json.loads(text[start : end + 1])
    states = obj.get("states")
    if not isinstance(states, list) or len(states) != 8:
        raise ValueError(f"states must be length 8, got {states!r}")
    cleaned = []
    for i, s in enumerate(states):
        try:
            v = int(s)
        except (TypeError, ValueError) as e:
            raise ValueError(f"states[{i}] not int: {s!r}") from e
        if v not in STATES:
            raise ValueError(f"states[{i}] out of range: {v}")
        cleaned.append(v)
    obj["states"] = cleaned
    obj["sentence_id"] = obj.get("sentence_id") or sentence_id
    obj.setdefault("evidence", {})
    obj.setdefault("exclude", False)
    obj.setdefault("needs_review", False)
    obj.setdefault("notes", "")
    # normalize evidence keys
    ev = obj["evidence"] if isinstance(obj["evidence"], dict) else {}
    obj["evidence"] = {a: str(ev.get(a, "") or "") for a in ASPECTS}
    return obj


def annotate_one(client: OpenAI, model: str, row: dict, max_retries: int = 3) -> dict:
    user = USER_TEMPLATE.format(
        sid=row["sentence_id"],
        sentence=row["sentence"],
        before=row.get("context_before") or "",
        after=row.get("context_after") or "",
    )
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            # Aspect labeling is simple classification — keep reasoning effort low.
            # deepseek-flash: disable thinking; json_object mode returns empty content.
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": PROMPT_SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=1200,
                extra_body={"thinking": {"type": "disabled"}},
            )
            if "flash" not in model and "reasoner" not in model:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            obj = parse_json_reply(content, row["sentence_id"])
            obj["model"] = model
            obj["source"] = "deepseek_ai_suggestion"
            obj["review_status"] = "ai_suggested"  # human must flip to confirmed
            obj["sentence"] = row["sentence"]
            return obj
        except Exception as e:  # noqa: BLE001 — retry then surface
            last_err = e
            wait = min(2**attempt, 8)
            print(f"  retry {attempt}/{max_retries} for {row['sentence_id']}: {e}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"Failed {row['sentence_id']}: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek-assisted aspect labeling")
    ap.add_argument("--start", type=int, default=0, help="start row index (0-based)")
    ap.add_argument("--limit", type=int, default=20, help="how many sentences to label")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument(
        "--out",
        default="labels_ai_suggestions.jsonl",
        help="output JSONL path (relative to this folder)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip sentence_ids already present in --out",
    )
    ap.add_argument(
        "--include-flagged",
        action="store_true",
        help="also annotate rows that look like hotel replies (default: skip)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the prompt for the first sentence and exit",
    )
    args = ap.parse_args()

    api_key = load_api_key()
    client = OpenAI(api_key=api_key, base_url=args.base_url)
    rows = load_sentences()
    print(f"Loaded {len(rows)} sentences. model={args.model} base={args.base_url}")

    out_path = ROOT / args.out
    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done_ids.add(json.loads(line)["sentence_id"])
                except Exception:  # noqa: BLE001
                    pass
        print(f"Resume: {len(done_ids)} already labeled in {out_path.name}")

    # default skip likely hotel-reply rows unless user wants them
    flagged = load_exclude_ids()
    if not flagged and RECORDS_TO_REVIEW.exists():
        with RECORDS_TO_REVIEW.open(encoding="utf-8-sig", newline="") as f:
            flagged = {r["sentence_id"] for r in csv.DictReader(f) if r.get("sentence_id")}

    batch = rows[args.start : args.start + args.limit]
    if not args.include_flagged:
        before_n = len(batch)
        batch = [r for r in batch if r["sentence_id"] not in flagged]
        skipped = before_n - len(batch)
        if skipped:
            print(f"Skipped {skipped} flagged reply-like rows (use --include-flagged to keep)")

    if args.dry_run:
        row = batch[0]
        print(USER_TEMPLATE.format(
            sid=row["sentence_id"],
            sentence=row["sentence"],
            before=row.get("context_before") or "",
            after=row.get("context_after") or "",
        ))
        return

    to_do = [r for r in batch if r["sentence_id"] not in done_ids]
    print(f"Will annotate {len(to_do)} sentences → {out_path.name}")

    mode = "a" if args.resume or out_path.exists() else "w"
    n_ok = n_fail = 0
    with out_path.open(mode, encoding="utf-8") as out:
        for i, row in enumerate(to_do, 1):
            sid = row["sentence_id"]
            print(f"[{i}/{len(to_do)}] {sid}: {row['sentence'][:40]}...")
            try:
                obj = annotate_one(client, args.model, row)
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"  FAIL {sid}: {e}", file=sys.stderr)
                continue
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out.flush()
            n_ok += 1
            states = obj["states"]
            mentioned = [
                f"{ASPECTS[j]}={states[j]}({STATES[states[j]]})"
                for j in range(8)
                if states[j] != 0
            ]
            print("  →", ", ".join(mentioned) if mentioned else "all absent")
            # gentle pacing; DeepSeek is usually fine without this
            time.sleep(0.2)

    print(f"\nDone. ok={n_ok} fail={n_fail} out={out_path}")
    print("Next: review every row, then export confirmed labels for training.")
    print("Suggested review fields: review_status → human_confirmed | needs_review | excluded")


if __name__ == "__main__":
    main()
