#!/usr/bin/env python3
"""Push the AI news digest to a Lark/Feishu custom bot webhook.

Reads data/latest-24h.json (already AI-filtered), picks top items by
ai_score, and posts an interactive card. Designed to run from GitHub
Actions; uses only Python stdlib.

Webhook URL is taken from the LARK_WEBHOOK env var. Optional
LARK_WEBHOOK_SECRET enables Lark's signature verification.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path


BEIJING_TZ = timezone(timedelta(hours=8))

LABEL_EMOJI = {
    "model_release": "🚀",
    "product_launch": "✨",
    "research": "🧪",
    "tool": "🛠",
    "company": "🏢",
    "policy": "⚖️",
    "opinion": "💬",
    "tutorial": "📘",
}


def load_items(data_path: Path) -> tuple[list[dict], dict]:
    with data_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    items = payload.get("items_ai") or payload.get("items") or []
    return items, payload


def pick_top(items: list[dict], limit: int) -> list[dict]:
    def sort_key(item: dict) -> tuple:
        return (
            float(item.get("ai_score") or 0.0),
            item.get("published_at") or "",
        )

    deduped: dict[str, dict] = {}
    for item in items:
        key = item.get("url") or item.get("id") or item.get("title")
        if not key or key in deduped:
            continue
        deduped[key] = item

    ordered = sorted(deduped.values(), key=sort_key, reverse=True)
    return ordered[:limit]


def format_item(item: dict) -> str:
    title = (
        item.get("title_bilingual")
        or item.get("title_zh")
        or item.get("title")
        or item.get("title_original")
        or "(无标题)"
    )
    url = item.get("url") or ""
    source = item.get("source") or item.get("site_name") or "未知来源"
    score = item.get("ai_score")
    label = item.get("ai_label") or ""
    emoji = LABEL_EMOJI.get(label, "•")

    parts = [source]
    if isinstance(score, (int, float)):
        parts.append(f"score {score:.2f}")
    if label:
        parts.append(label)
    meta = " · ".join(parts)

    safe_title = title.replace("[", "(").replace("]", ")")
    return f"{emoji} **[{safe_title}]({url})**\n   <font color='grey'>{meta}</font>"


def build_card(items: list[dict], payload: dict, limit: int) -> dict:
    generated_at = payload.get("generated_at") or ""
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        ts_local = ts.astimezone(BEIJING_TZ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        ts_local = generated_at or datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M")

    total = payload.get("total_items", len(items))
    sources = payload.get("source_count", "?")
    sites = payload.get("site_count", "?")

    subtitle = (
        f"近 24h 共 {total} 条 AI 强相关 · {sites} 站点 / {sources} 信源"
        f" · 快照 {ts_local}"
    )

    elements: list[dict] = [
        {
            "tag": "div",
            "text": {"tag": "lark_md", "content": subtitle},
        },
        {"tag": "hr"},
    ]

    body = "\n\n".join(format_item(it) for it in items)
    elements.append({"tag": "div", "text": {"tag": "lark_md", "content": body}})

    elements.append({"tag": "hr"})
    elements.append(
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "查看完整网页"},
                    "type": "primary",
                    "url": "https://learnprompt.github.io/ai-news-radar/",
                }
            ],
        }
    )

    return {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"🤖 24h AI 雷达 Top {len(items)}",
                },
                "template": "blue",
            },
            "elements": elements,
        },
    }


def sign(secret: str, timestamp: int) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(
        string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(digest).decode("utf-8")


def post(webhook: str, body: dict, secret: str | None) -> dict:
    if secret:
        ts = int(time.time())
        body = {**body, "timestamp": str(ts), "sign": sign(secret, ts)}
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8", "replace")
                return json.loads(raw) if raw else {}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Lark webhook failed after retries: {last_err}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/latest-24h.json",
        help="Path to latest-24h.json (default: data/latest-24h.json)",
    )
    parser.add_argument(
        "--limit", type=int, default=15, help="Max items in the digest (default: 15)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the card body to stdout instead of posting",
    )
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"error: data file not found: {data_path}", file=sys.stderr)
        return 2

    items, payload = load_items(data_path)
    if not items:
        print("warning: no AI items to push; exiting without sending.", file=sys.stderr)
        return 0

    top = pick_top(items, args.limit)
    card = build_card(top, payload, args.limit)

    if args.dry_run:
        json.dump(card, sys.stdout, ensure_ascii=False, indent=2)
        print()
        return 0

    webhook = os.environ.get("LARK_WEBHOOK", "").strip()
    if not webhook:
        print("error: LARK_WEBHOOK env var is empty", file=sys.stderr)
        return 3

    secret = os.environ.get("LARK_WEBHOOK_SECRET", "").strip() or None
    resp = post(webhook, card, secret)
    code = resp.get("code", resp.get("StatusCode"))
    if code not in (0, None):
        print(f"error: lark webhook returned {resp}", file=sys.stderr)
        return 4
    print(f"sent {len(top)} items to Lark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
