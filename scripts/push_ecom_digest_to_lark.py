#!/usr/bin/env python3
"""Push an ecommerce-focused AI digest to a Lark/Feishu webhook.

Reads data/latest-24h.json, takes the top N AI items, then asks Claude to
distill them into 5-8 actionable opportunities for a Chinese multi-platform
ecommerce operator (天猫 / 拼多多 / 抖音 / 小红书 / 得物 / 京东).

Output items use the format:
    1. **<bilingual title>**
       **🔍 机会**：...
       **⚠️ 需确认**：...   (optional)
       **💡 落地**：...
       **🔗 来源**：[source](url)

The card is posted via the same custom webhook used by push_to_lark.py.
Requires:
  LARK_WEBHOOK              — webhook URL
  LARK_WEBHOOK_SECRET       — optional signing secret
  ANTHROPIC_BASE_URL        — Claude-compatible API base URL
  ANTHROPIC_AUTH_TOKEN      — bearer / api key for that endpoint
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

import anthropic


BEIJING_TZ = timezone(timedelta(hours=8))

MODEL = "claude-haiku-4-5"
INPUT_TOP_N = 40
TARGET_ITEMS = "5-8"

SYSTEM_PROMPT = """你是一位资深电商运营顾问，服务对象是一位在中国做多平台电商的商家。

商家在售平台：天猫、拼多多、抖音、小红书、得物、京东。
他日常关心的事：选品、商品主图与详情页、短视频与图文种草、直播带货、客服话术、投流（千川/万相台/聚星/京准通）、评价管理、GMV 与转化率提升、平台规则与算法变动、消费者行为趋势、AI 工具在电商中的实际应用。

你的任务：阅读用户给你的近 24 小时 AI 行业资讯列表，从中筛选 5-8 条对他真正有用的，并按以下严格格式输出（不要加任何前后缀、解释、客套话或结尾总结）：

1. **<标题（保留中英双语，如果原标题是英文，给一句中文意译）>**
   【机会】<一两句话讲清楚：这条新闻对中国电商商家意味着什么机会或威胁，要具体到品类/平台/环节，不要空话>
   【需确认】<可选：如果有需要核实的关键事实或前提条件，简短列出；没有就整行省略>
   【落地】<给出 1 条今天/本周就能动手的具体动作，要写到平台名 + 具体场景，例如"在抖音用 X 工具把现有 30 条服饰短视频批量改成竖版口播版本"，禁止空话如"关注趋势""调整策略">
   【来源】[<原文标题或来源名>](<URL>)

2. **...**

筛选与写作规则：
- 只挑对中国电商运营真正有用的，与电商场景无关的纯学术、纯硅谷融资八卦、海外消费者新闻等一律跳过。
- 同一类话题只保留信息量最高的一条，避免重复。
- 优先选：AI 生成图/视频、数字人/口播、智能客服、智能投放、商品理解/搜索、平台官方 AI 功能、消费者用 AI 购物、低成本批量内容生产。
- 中文输出。条数控制在 5-8 条。如果当天确实没有任何条目对电商有用，只输出一行：`今日无电商相关高价值线索。`
"""


def load_items(data_path: Path) -> list[dict]:
    with data_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload.get("items_ai") or payload.get("items") or []


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


def render_input_block(items: list[dict]) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        title = (
            it.get("title_bilingual")
            or it.get("title_zh")
            or it.get("title")
            or it.get("title_original")
            or "(无标题)"
        )
        url = it.get("url") or ""
        source = it.get("source") or it.get("site_name") or "未知"
        label = it.get("ai_label") or ""
        reason = it.get("ai_relevance_reason") or ""
        meta = f"来源：{source}"
        if label:
            meta += f" · 类型：{label}"
        line = f"[{i}] {title}\n    URL: {url}\n    {meta}"
        if reason:
            line += f"\n    背景：{reason}"
        lines.append(line)
    return "\n\n".join(lines)


def call_claude(items: list[dict]) -> str:
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip() or None
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip() or os.environ.get(
        "ANTHROPIC_API_KEY", ""
    ).strip()
    if not token:
        raise RuntimeError("ANTHROPIC_AUTH_TOKEN (or ANTHROPIC_API_KEY) is required")

    client = anthropic.Anthropic(
        api_key=token,
        base_url=base_url,
        default_headers={"Authorization": f"Bearer {token}"},
    )

    user_block = (
        f"以下是最近 24 小时挑出来的 {len(items)} 条 AI 行业资讯，请按系统提示的格式输出：\n\n"
        + render_input_block(items)
    )

    with client.messages.stream(
        model=MODEL,
        max_tokens=3000,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_block}],
    ) as stream:
        final = stream.get_final_message()

    parts = [b.text for b in final.content if getattr(b, "type", None) == "text"]
    return "".join(parts).strip()


def beautify(raw: str) -> str:
    return (
        raw.replace("【机会】", "**🔍 机会**：")
        .replace("【需确认】", "**⚠️ 需确认**：")
        .replace("【落地】", "**💡 落地**：")
        .replace("【来源】", "**🔗 来源**：")
    )


def build_card(body_md: str) -> dict:
    today = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d")
    elements = [
        {
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": "**为你筛选的电商可落地线索**（基于近 24h AI 资讯，由 Claude 生成）",
            },
        },
        {"tag": "hr"},
        {"tag": "div", "text": {"tag": "lark_md", "content": body_md}},
        {"tag": "hr"},
        {
            "tag": "action",
            "actions": [
                {
                    "tag": "button",
                    "text": {"tag": "plain_text", "content": "查看完整 AI 雷达"},
                    "type": "default",
                    "url": "https://learnprompt.github.io/ai-news-radar/",
                }
            ],
        },
    ]
    return {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {
                    "tag": "plain_text",
                    "content": f"🛒 AI 电商机会日报 ｜ {today}",
                },
                "template": "orange",
            },
            "elements": elements,
        },
    }


def sign(secret: str, timestamp: int) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
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
    parser.add_argument("--data", default="data/latest-24h.json")
    parser.add_argument("--input-top", type=int, default=INPUT_TOP_N)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated card to stdout instead of posting",
    )
    parser.add_argument(
        "--print-llm",
        action="store_true",
        help="Also print the raw LLM output before posting",
    )
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"error: data file not found: {data_path}", file=sys.stderr)
        return 2

    items = load_items(data_path)
    if not items:
        print("warning: no AI items; skipping ecom digest.", file=sys.stderr)
        return 0

    top = pick_top(items, args.input_top)
    print(f"feeding {len(top)} items to {MODEL}...", file=sys.stderr)

    raw = call_claude(top)
    if not raw:
        print("error: empty response from Claude", file=sys.stderr)
        return 5

    if args.print_llm:
        print("--- raw LLM output ---", file=sys.stderr)
        print(raw, file=sys.stderr)
        print("--- end ---", file=sys.stderr)

    body_md = beautify(raw)
    card = build_card(body_md)

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
    print("ecom digest sent to Lark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
