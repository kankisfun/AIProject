#!/usr/bin/env python3
"""
Princess Bubblegum chat (Together.ai, OpenAI-compatible)

Setup:
  pip install openai python-dotenv requests
  # Then put your key in .env as TOGETHER_API_KEY

Notes:
- Uses tool/function calling to fetch a tiny "factlet" sometimes.
- Enforces JSON output via response_format; falls back to JSON mode if needed.
- Keeps PB strictly adult, canon-ish, and SFW by default.

Docs used:
- OpenAI-compatible client with Together base_url: https://docs.together.ai/docs/openai-api-compatibility
- Function/tool calling on Together: https://docs.together.ai/docs/function-calling
- JSON/structured outputs: https://docs.together.ai/docs/json-mode (OpenAI-style response_format)
- Model name: meta-llama/Llama-3.3-70B-Instruct-Turbo
"""
import os
import json
import random
import re
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

# ---------- env & client ----------

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
BASE_URL = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
MODEL = os.getenv("TOGETHER_CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")

FACT_FREQ = float(os.getenv("FACT_FETCH_FREQUENCY", "0.3"))
ALLOW_WEB_FACTS = os.getenv("ALLOW_WEB_FACTS", "true").lower() == "true"

if not TOGETHER_API_KEY:
    raise SystemExit("Missing TOGETHER_API_KEY in .env")

client = OpenAI(api_key=TOGETHER_API_KEY, base_url=BASE_URL)

# ---------- persona card (compact) ----------

# Your longer description, distilled to stay cheap in tokens but true to the vibe.
PB_PERSONA = (
    "You are Princess Bubblegum (PB), the adult sovereign scientist of the Candy Kingdom. "
    "Blend royal authority with rigorous intellect and duty-first pragmatism. "
    "Warm to friends yet coolly rational when ruling; playful at times but serious when stakes rise. "
    "Long-lived, historically literate, and inventive—favor chemistry/engineering metaphors. "
    "Guard vulnerabilities behind professionalism; make hard, sometimes gray choices to protect your people. "
    "Stay canon when possible; if unsure, say so briefly. Keep replies to 1–3 sentences, no camera/LoRA talk. "
    "Safety: no minors, no non-consensual content, no explicit sexual content."
)

# ---------- tool spec (OpenAI-style) ----------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_random_fact",
            "description": (
                "Return a short, canon-friendly fact about Princess Bubblegum or Adventure Time, "
                "with a citation. Keep it SFW and clearly adult-context for PB."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Optional focus, e.g., 'science', 'politics', 'relationships'."
                    }
                },
                "required": []
            },
        },
    }
]

# Structured output schema (OpenAI response_format style)
RESPONSE_FORMAT_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "PBReply",
        "strict": True,  # ask for strict JSON adherence
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "line": {"type": "string"},
                "did_use_fact": {"type": "boolean"},
                "fact": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "source_title": {"type": "string"},
                        "source_url": {"type": "string"},
                        "year": {"type": "integer"},
                    },
                    "required": ["text", "source_title", "source_url", "year"]
                },
            },
            "required": ["line", "did_use_fact"]
        },
    },
}

# Fallback JSON-only mode if json_schema isn't supported by the server/model
RESPONSE_FORMAT_JSON_ONLY = {"type": "json_object"}


# ---------- fact fetching (seed + optional web) ----------

def seed_facts() -> List[Dict[str, Any]]:
    """Safe, tiny seed set so the tool works offline."""
    return [
        {
            "text": "Princess Bubblegum rules the Candy Kingdom and is renowned for her scientific expertise.",
            "source_title": "Princess Bubblegum — Overview",
            "source_url": "https://en.wikipedia.org/wiki/Princess_Bubblegum",
            "year": 2010,
        },
        {
            "text": "PB often balances compassion for her citizens with pragmatic decisions required of a ruler.",
            "source_title": "Series Themes",
            "source_url": "https://en.wikipedia.org/wiki/Adventure_Time",
            "year": 2010,
        },
        {
            "text": "She has created numerous inventions that shape daily life in the Candy Kingdom.",
            "source_title": "Character Inventions",
            "source_url": "https://adventuretime.fandom.com/wiki/Princess_Bubblegum",
            "year": 2010,
        },
    ]


def is_safe_fact(text: str) -> bool:
    """Very light SFW/age screening."""
    text_l = text.lower()
    banned = ["minor", "underage", "loli", "shota"]
    return not any(b in text_l for b in banned)


def sentences(s: str) -> List[str]:
    # very rough sentence split to keep deps minimal
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    return [p.strip() for p in parts if 10 <= len(p.strip()) <= 220]


def web_wikipedia_fact() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/Princess_Bubblegum",
            timeout=6,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        lines = sentences(data.get("extract", ""))
        if not lines:
            return None
        line = random.choice(lines)
        if not is_safe_fact(line):
            return None
        return {
            "text": line,
            "source_title": data.get("title", "Wikipedia"),
            "source_url": data.get("content_urls", {}).get("desktop", {}).get("page", "https://wikipedia.org"),
            "year": 2010,
        }
    except Exception:
        return None


def web_fandom_fact() -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(
            "https://adventuretime.fandom.com/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": "Princess_Bubblegum",
                "format": "json",
            },
            timeout=8,
        )
        if r.status_code != 200:
            return None
        j = r.json()
        pages = j.get("query", {}).get("pages", {})
        if not pages:
            return None
        page = next(iter(pages.values()))
        extract = page.get("extract", "")
        lines = sentences(extract)
        if not lines:
            return None
        line = random.choice(lines)
        if not is_safe_fact(line):
            return None
        return {
            "text": line,
            "source_title": "Adventure Time Wiki (Fandom)",
            "source_url": "https://adventuretime.fandom.com/wiki/Princess_Bubblegum",
            "year": 2010,
        }
    except Exception:
        return None


def fetch_random_fact(topic: Optional[str] = None) -> Dict[str, Any]:
    """Combine seeds with optional live fetch; return one fact dict."""
    pool = seed_facts()
    if ALLOW_WEB_FACTS:
        for getter in (web_wikipedia_fact, web_fandom_fact):
            fact = getter()
            if fact:
                pool.append(fact)
    # naive topic bias
    if topic:
        topic_l = topic.lower()
        weighted = [f for f in pool if topic_l in f["text"].lower()]
        if weighted:
            pool = weighted
    # ensure safety
    pool = [f for f in pool if is_safe_fact(f["text"])]
    return random.choice(pool if pool else seed_facts())


# ---------- chat core ----------

def safe_json_loads(maybe_json: str) -> Dict[str, Any]:
    """Parse JSON; if it fails, try to extract the first {...} block."""
    try:
        return json.loads(maybe_json)
    except Exception:
        m = re.search(r"\{.*\}", maybe_json, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    # final fallback: wrap as a minimal object
    return {"line": maybe_json.strip(), "did_use_fact": False}


def complete(messages: List[Dict[str, str]], allow_tool: bool) -> Dict[str, Any]:
    """One full turn: may include tool call + follow-up for final JSON."""
    # First request
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "response_format": RESPONSE_FORMAT_JSON_SCHEMA,
    }

    if allow_tool:
        kwargs["tools"] = TOOLS
        kwargs["tool_choice"] = "auto"
    else:
        kwargs["tool_choice"] = "none"

    try:
        res = client.chat.completions.create(**kwargs)
    except Exception as e:
        # Some servers/models only support JSON mode; retry
        kwargs["response_format"] = RESPONSE_FORMAT_JSON_ONLY
        res = client.chat.completions.create(**kwargs)

    msg = res.choices[0].message

    # If the model wants to call our tool, execute it and finalize
    if allow_tool and getattr(msg, "tool_calls", None):
        for tc in msg.tool_calls:
            if tc.function.name == "fetch_random_fact":
                args = {}
                if tc.function.arguments:
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                fact = fetch_random_fact(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(fact),
                })

        # Ask for the final structured message (no more tools)
        try:
            res2 = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                response_format=RESPONSE_FORMAT_JSON_SCHEMA,
                tool_choice="none",
            )
        except Exception:
            res2 = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                response_format=RESPONSE_FORMAT_JSON_ONLY,
                tool_choice="none",
            )
        msg = res2.choices[0].message

    # Parse final JSON payload
    return safe_json_loads(msg.content)


def chat_once(user_text: str) -> Dict[str, Any]:
    # Randomly enable tool usage, to keep the surprises occasional & cheap
    allow_tool = (random.random() < FACT_FREQ)
    messages = [
        {"role": "system", "content": PB_PERSONA},
        {"role": "user", "content": user_text},
    ]
    reply = complete(messages, allow_tool=allow_tool)
    # Ensure required fields
    reply.setdefault("did_use_fact", False)
    if reply.get("did_use_fact") and "fact" not in reply:
        reply["did_use_fact"] = False
    return reply


# ---------- demo REPL ----------

if __name__ == "__main__":
    print("💬 Chatting with Princess Bubblegum. Press Ctrl+C to quit.")
    print("Tip: Ask things like 'How was court today?' or 'Propose a science date idea.'\n")
    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            out = chat_once(user)
            print(f"PB: {out.get('line','(no line)')}")
            if out.get("did_use_fact") and isinstance(out.get("fact"), dict):
                f = out["fact"]
                print(f"   (Fun fact: {f.get('text')} — {f.get('source_title')})")
    except KeyboardInterrupt:
        print("\nBye!")
