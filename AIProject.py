#!/usr/bin/env python3
"""
Princess Bubblegum chat (Together.ai, OpenAI-compatible)

Setup:
  pip install openai python-dotenv
  # Then put your key in .env as TOGETHER_API_KEY

Notes:
- Enforces JSON output via response_format; falls back to JSON mode if needed.
- Keeps PB strictly adult, canon-ish, and SFW by default.

Docs used:
- OpenAI-compatible client with Together base_url: https://docs.together.ai/docs/openai-api-compatibility
- JSON/structured outputs: https://docs.together.ai/docs/json-mode (OpenAI-style response_format)
- Model name: meta-llama/Llama-3.3-70B-Instruct-Turbo
"""
import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List

import requests
from dotenv import load_dotenv
from openai import OpenAI

# ---------- env & client ----------

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
BASE_URL = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
MODEL = os.getenv("TOGETHER_CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")

if not TOGETHER_API_KEY:
    raise SystemExit("Missing TOGETHER_API_KEY in .env")

client = OpenAI(api_key=TOGETHER_API_KEY, base_url=BASE_URL)

# ---------- lora tag categorization ----------

LORA_REGISTRY_PATH = os.path.normpath(
    os.getenv(
        "LORA_REGISTRY_PATH",
        r"E:\\ConfyUI_New\\ComfyUI\\models\\loras\\lora_registry.json",
    )
)
BUBBLEGUM_LORA_NAME = "Bubblegum_ILL.safetensors"
BUBBLEGUM_JSON_PATH = os.path.join(
    os.path.dirname(LORA_REGISTRY_PATH), "Bubblegum_ILL.json"
)

IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

BASE_POSITIVE = "high quality, 1girl, sexy"
BASE_NEGATIVE = (
    "low quality, jpeg artifacts, deformed, extra fingers, text, watermark, logo, censored"
)

CATEGORY_KEYS = [
    "hairstyle_hat_head_toppings",
    "expressions",
    "fullbody_clothes",
    "up_clothes",
    "bottom_clothes",
    "accessories",
    "specific_body_parts",
    "skin_fur",
    "position_sex_position",
    "camera_framing",
    "background",
    "other",
    "unknown",
]


def chunked(seq: List[Any], size: int) -> List[List[Any]]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def categorize_tags_with_ai(tags: List[str]) -> Dict[str, List[str]]:
    """Use the chat model to bucket tags into preset categories."""
    categorized = {k: [] for k in CATEGORY_KEYS}
    system_msg = (
        "You sort character description tags into predefined categories. "
        "Return JSON with keys exactly: "
        + ", ".join(CATEGORY_KEYS)
        + ". Use 'unknown' for tags you cannot place."
    )
    chunks = chunked(tags, 30)
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        print(f"[LoRA] Categorizing tag chunk {idx}/{total}...")
        try:
            res = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_msg},
                    {
                        "role": "user",
                        "content": (
                            "Categorize these tags into the categories. Reply with JSON only.\nTags: "
                            + json.dumps(chunk)
                        ),
                    },
                ],
                response_format={"type": "json_object"},
            )
            data = json.loads(res.choices[0].message.content)
            for key in CATEGORY_KEYS:
                if isinstance(data.get(key), list):
                    categorized[key].extend(data[key])
        except Exception as e:
            print(f"[LoRA] Warning: AI categorization failed on chunk {idx}: {e}")
    return categorized


def ensure_bubblegum_tags_file() -> None:
    """Create the Bubblegum tag categorization file once."""
    print(f"[LoRA] Ensuring tag file at {BUBBLEGUM_JSON_PATH}")
    if os.path.exists(BUBBLEGUM_JSON_PATH):
        print("[LoRA] Tag file already exists; skipping categorization.")
        return
    try:
        print(f"[LoRA] Loading registry from {LORA_REGISTRY_PATH}")
        with open(LORA_REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
        # Some registry formats nest entries under a top-level "loras" key.
        if isinstance(registry, dict):
            loras = registry.get("loras", registry)
        else:
            loras = {}
        tags = loras.get(BUBBLEGUM_LORA_NAME, {}).get("tags", [])
        print(f"[LoRA] Found {len(tags)} tags for {BUBBLEGUM_LORA_NAME}")
        if not tags:
            print("[LoRA] No tags found; skipping file creation.")
            return
        categorized = categorize_tags_with_ai(tags)
        with open(BUBBLEGUM_JSON_PATH, "w", encoding="utf-8") as out:
            json.dump(categorized, out, indent=2)
        print(f"[LoRA] Wrote categorized tags to {BUBBLEGUM_JSON_PATH}")
    except Exception as e:
        print(f"[LoRA] Warning: could not create Bubblegum tag file: {e}")


def sanitize(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_")


def next_run_number(images_dir: Path, prefix: str) -> int:
    pattern = re.compile(rf"{re.escape(prefix)}__run(\d+)")
    max_n = -1
    for p in images_dir.glob(f"{prefix}__run*"):
        m = pattern.search(p.stem)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def assemble_and_send_prompt() -> None:
    """Pick tags via AI, build a ComfyUI workflow, and POST it."""
    try:
        with open(BUBBLEGUM_JSON_PATH, "r", encoding="utf-8") as f:
            categorized = json.load(f)
    except Exception as e:
        print(f"[Prompt] Could not load categorized tags: {e}")
        return

    system_msg = (
        "You build outfit prompts for an image generator. "
        "From the provided options choose: 2 hairstyle tags, 1 expression, "
        "1 fullbody outfit OR 1 top and 1 bottom, 1 accessory, 1 specific body part, "
        "1 skin type, 1 position, and 1 background. "
        "Return JSON with only the selected tags in their respective fields; no extra text."
    )
    user_msg = "Options:\n" + json.dumps(
        {
            "hairstyle_hat_head_toppings": categorized.get(
                "hairstyle_hat_head_toppings", []
            ),
            "expressions": categorized.get("expressions", []),
            "fullbody_clothes": categorized.get("fullbody_clothes", []),
            "up_clothes": categorized.get("up_clothes", []),
            "bottom_clothes": categorized.get("bottom_clothes", []),
            "accessories": categorized.get("accessories", []),
            "specific_body_parts": categorized.get("specific_body_parts", []),
            "skin_fur": categorized.get("skin_fur", []),
            "position_sex_position": categorized.get("position_sex_position", []),
            "background": categorized.get("background", []),
        }
    )
    try:
        res = client.chat.completions.create(
            model=MODEL,
            temperature=1,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        choice = json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"[Prompt] AI tag selection failed: {e}")
        return

    selected_tags: List[str] = []
    selected_tags.extend(choice.get("hairstyle_hat_head_toppings", [])[:2])
    if exp := choice.get("expressions"):
        selected_tags.append(exp)
    if choice.get("fullbody_clothes"):
        selected_tags.append(choice["fullbody_clothes"])
    else:
        if top := choice.get("up_clothes"):
            selected_tags.append(top)
        if bottom := choice.get("bottom_clothes"):
            selected_tags.append(bottom)
    for key in [
        "accessories",
        "specific_body_parts",
        "skin_fur",
        "position_sex_position",
        "background",
    ]:
        val = choice.get(key)
        if val:
            selected_tags.append(val)

    print(f"[Prompt] {json.dumps(selected_tags)}")

    try:
        with open(LORA_REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
        loras = registry.get("loras", registry)
        entry = loras.get(BUBBLEGUM_LORA_NAME, {})
    except Exception:
        entry = {}
    lora_path = entry.get("path", BUBBLEGUM_LORA_NAME).replace("\\", "/")

    # [FIX 1] Normalize trigger that may be str or list; pick first token safely
    trigger_raw = entry.get("trigger") or entry.get("trigger_words") or ""
    trigger_text = ", ".join(trigger_raw) if isinstance(trigger_raw, list) else str(trigger_raw)
    token = (trigger_text.split() or [os.path.splitext(BUBBLEGUM_LORA_NAME)[0]])[0]

    base_pos = BASE_POSITIVE
    if trigger_text:
        base_pos += ", " + trigger_text

    # [FIX 2] Flatten and stringify tags so join never sees lists
    selected_tags = [
        str(x).strip()
        for item in selected_tags
        for x in (item if isinstance(item, list) else [item])
        if str(x).strip()
    ]

    positive_text = base_pos + (", " + ", ".join(selected_tags) if selected_tags else "")
    print(f"[Prompt] Positive prompt: {positive_text}")

    try:
        with open("1lora.json", "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        print(f"[Prompt] Could not load workflow template: {e}")
        return

    workflow.get("6", {}).setdefault("inputs", {})["text"] = positive_text
    workflow.get("7", {}).setdefault("inputs", {})["text"] = BASE_NEGATIVE
    lora_inputs = workflow.get("10", {}).setdefault("inputs", {})
    lora_inputs["lora_name"] = lora_path
    lora_inputs["strength_model"] = 0.75
    lora_inputs["strength_clip"] = 0.75
    workflow.get("3", {}).setdefault("inputs", {})["seed"] = 504
    workflow.get("4", {}).setdefault("inputs", {})["ckpt_name"] = (
        "waiNSFWIllustrious_v120.safetensors"
    )
    workflow.get("12", {}).setdefault("inputs", {})["model_name"] = (
        "RealESRGAN_x4plus.pth"
    )

    short_slug = "__".join(sanitize(t) for t in selected_tags[:3])
    prefix_base = f"{sanitize(token)}__{short_slug}"
    n = next_run_number(IMAGES_DIR, prefix_base)
    filename_prefix = f"{IMAGES_DIR.name}/{prefix_base}__run{n}"
    workflow.get("9", {}).setdefault("inputs", {})["filename_prefix"] = filename_prefix

    try:
        r = requests.post(
            "http://127.0.0.1:8188/prompt", json={"prompt": workflow}, timeout=10
        )
        print(f"[ComfyUI] POST status {r.status_code}")
    except Exception as e:
        print(f"[ComfyUI] Warning: {e}")

# ---------- persona card (compact) ----------

# Your longer description, distilled to stay cheap in tokens but true to the vibe.
PB_PERSONA = (
    "You are Princess Bubblegum (PB), the adult sovereign scientist of the Candy Kingdom. "
    "Blend royal authority with rigorous intellect and duty-first pragmatism. "
    "Warm to friends yet coolly rational when ruling; playful at times but serious when stakes rise. "
    "Long-lived, historically literate, and inventive—favor chemistry/engineering metaphors. "
    "Guard vulnerabilities behind professionalism; make hard, sometimes gray choices to protect your people. "
    "Stay canon when possible; if unsure, say so briefly. Keep replies to 1–3 sentences, no camera talk. "
)

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
            },
            "required": ["line"],
        },
    },
}

# Fallback JSON-only mode if json_schema isn't supported by the server/model
RESPONSE_FORMAT_JSON_ONLY = {"type": "json_object"}


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
    return {"line": maybe_json.strip()}


def complete(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """One full turn returning structured JSON."""
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            response_format=RESPONSE_FORMAT_JSON_SCHEMA,
        )
    except Exception:
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            response_format=RESPONSE_FORMAT_JSON_ONLY,
        )

    msg = res.choices[0].message

    # Parse final JSON payload
    return safe_json_loads(msg.content)


def chat_once(user_text: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": PB_PERSONA},
        {"role": "user", "content": user_text},
    ]
    return complete(messages)


# ---------- demo REPL ----------

if __name__ == "__main__":
    ensure_bubblegum_tags_file()
    try:
        assemble_and_send_prompt()
    except Exception as e:
        print(f"[ComfyUI] Prompt assembly failed: {e}")
    print("💬 Chatting with Princess Bubblegum. Press Ctrl+C to quit.")
    print("Tip: Ask things like 'How was court today?' or 'Propose a science date idea.'\n")
    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            out = chat_once(user)
            print(f"PB: {out.get('line','(no line)')}")
    except KeyboardInterrupt:
        print("\nBye!")
