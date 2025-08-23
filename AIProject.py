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
import sys
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

import tkinter as tk
from PIL import Image, ImageTk
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

# NEW: where ComfyUI actually writes images (default)
COMFY_OUTPUT_DIR = Path(os.getenv("COMFY_OUTPUT_DIR", r"E:\ConfyUI_New\ComfyUI\output"))
WATCH_DIR = COMFY_OUTPUT_DIR / IMAGES_DIR.name  # e.g., .../ComfyUI/output/images
WATCH_DIR.mkdir(parents=True, exist_ok=True)


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
    "background",
    "other",
    "unknown",
]

EXTRA_EXPRESSIONS = [
    "Neutral face",
    "Gentle smile",
    "Wide open smile happy",
    "Closed eyes smile joyful",
    "Laughing with open mouth squinting eyes",
    "Small frown slight sadness",
    "Big frown serious sadness",
    "Crying with tears streaming",
    "Pouting with cheeks puffed",
    "Blushing smile shy happy",
    "Blushing looking away embarrassed",
    "Angry with teeth showing",
    "Angry shouting yelling",
    "Confused head tilted raised eyebrow",
    "Worried with sweat drop",
    "Surprised wide eyes open mouth",
    "Shocked pale face wide eyes",
    "Determined sharp eyes set mouth",
    "Smirk smug half smile",
    "Mischievous teasing grin",
    "Sleepy half closed eyes yawning",
    "Serious flat expression sharp eyes",
    "Nervous laugh awkward sweat drop",
    "Daydreaming sparkly eyes soft smile",
    "Scared trembling mouth wide eyes",
    "Excited sparkling eyes wide grin",
    "Disgusted wrinkled nose frown",
    "Surprised blush shock with blush",
    "Thinking hand on chin furrowed brow",
    "Deadpan flat stare no emotion",
]

EXTRA_POSITIONS = [
    "Standing neutral arms at sides",
    "Standing one hand on hip",
    "Standing arms crossed",
    "Standing hands behind back",
    "Standing hands in pockets",
    "Standing arms raised victory pose",
    "Standing arms spread welcoming",
    "Standing leaning forward curious",
    "Sitting normal on chair",
    "Sitting casual slouched",
    "Sitting cross legged",
    "Sitting knees together hands on lap",
    "Sitting on ground legs stretched",
    "Kneeling basic pose",
    "Kneeling one knee up proposing stance",
    "Walking forward casual",
    "Running basic pose",
    "Jumping arms up cheerful",
    "Jumping sideways action pose",
    "Leaning against wall arms crossed",
    "Leaning forward on desk",
    "Reaching hand forward",
    "Pointing finger outward",
    "Peace sign V pose",
    "Waving hand friendly",
    "Hands clasped in front shy pose",
    "Hands behind head relaxed",
    "Arms wrapped around self nervous",
    "One hand out offering something",
    "Power up stance feet apart fists clenched",
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
            messages = [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": (
                        "Categorize these tags into the categories. Reply with JSON only.\nTags: "
                        + json.dumps(chunk)
                    ),
                },
            ]
            print("[AI Request]", json.dumps(messages, indent=2))
            res = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                messages=messages,
                response_format={"type": "json_object"},
            )
            print("[AI Response]", res.choices[0].message.content)
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


def choose_initial_tags(categorized: Dict[str, List[str]]) -> Dict[str, Any]:
    """Ask the model to pick initial tags from available options."""
    system_msg = (
        "You build outfit prompts for an image generator. "
        "From the provided options choose: 2 hairstyle tags, 1 expression, "
        "1 fullbody outfit OR 1 top and 1 bottom, 1 accessory, 1 specific body part, "
        "1 skin type, 1 position, and 1 background. "
        "Return JSON with only the selected tags in their respective fields; no extra text."
    )
    user_msg = "Options:\n" + json.dumps(
        {
            "hairstyle_hat_head_toppings": categorized.get("hairstyle_hat_head_toppings", []),
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
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        print("[AI Request]", json.dumps(messages, indent=2))
        res = client.chat.completions.create(
            model=MODEL,
            temperature=1,
            messages=messages,
            response_format={"type": "json_object"},
        )
        print("[AI Response]", res.choices[0].message.content)
        choice = json.loads(res.choices[0].message.content)
    except Exception as e:
        print(f"[Prompt] AI tag selection failed: {e}")
        return {}
    return choice


def assemble_and_send_prompt(choice: Dict[str, Any]) -> Path | None:
    """Build a ComfyUI workflow from chosen tags and POST it. Wait for the image."""
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

    try:
        with open(LORA_REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
        loras = registry.get("loras", registry)
        entry = loras.get(BUBBLEGUM_LORA_NAME, {})
    except Exception:
        entry = {}
    lora_path = entry.get("path", BUBBLEGUM_LORA_NAME).replace("\\", "/")

    trigger_raw = entry.get("trigger") or entry.get("trigger_words") or ""
    trigger_text = ", ".join(trigger_raw) if isinstance(trigger_raw, list) else str(trigger_raw)
    token = (trigger_text.split() or [os.path.splitext(BUBBLEGUM_LORA_NAME)[0]])[0]

    base_pos = BASE_POSITIVE
    if trigger_text:
        base_pos += ", " + trigger_text

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
        return None

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
    expected_prefix = Path(filename_prefix).name  # e.g. "images\pb__foo__run7" -> "pb__foo__run7"


    existing = set(WATCH_DIR.glob(f"{expected_prefix}*.png"))

    payload = {"prompt": workflow}
    print("[ComfyUI] Workflow payload:", json.dumps(payload, indent=2))
    try:
        r = requests.post(
            "http://127.0.0.1:8188/prompt", json=payload, timeout=10
        )
        print(f"[ComfyUI] POST status {r.status_code}")
        print(f"[ComfyUI] Response body: {r.text}")
    except Exception as e:
        print(f"[ComfyUI] Warning: {e}")
        return None

    for _ in range(240):  # up to ~2 minutes at 0.5s
        new_files = set(WATCH_DIR.glob(f"{expected_prefix}*.png")) - existing
        if new_files:
            candidate = max(new_files, key=os.path.getmtime)

            # wait until size is stable and image opens cleanly
            stable = 0
            last_size = -1
            for _ in range(20):  # up to ~10s extra
                try:
                    sz = candidate.stat().st_size
                    if sz == last_size:
                        # try to open/verify only once size stayed the same
                        from PIL import Image
                        with Image.open(candidate) as im:
                            im.verify()  # ensure not truncated
                        return candidate
                    last_size = sz
                    time.sleep(0.5)
                    stable += 1
                except Exception:
                    time.sleep(0.5)
        time.sleep(0.5)
    print("[ComfyUI] Warning: image not found or not complete in time")
    return None

# ---------- persona card (compact) ----------

# Your longer description, distilled to stay cheap in tokens but true to the vibe.
PB_PERSONA = (
    "You are Princess Bubblegum (PB), the adult sovereign scientist of the Candy Kingdom. "
    "Blend royal authority with rigorous intellect and duty-first pragmatism. "
    "Warm to friends yet coolly rational when ruling; playful at times but serious when stakes rise. "
    "Long-lived, historically literate, and inventive—favor chemistry/engineering metaphors. "
    "Guard vulnerabilities behind professionalism; make hard, sometimes gray choices to protect your people. "
    "Stay canon when possible; if unsure, say so briefly. Keep replies to 1–3 sentences, no camera talk. "
    "Sprinkle in brief stage directions for PB's actions wrapped in asterisks, e.g., *smirk* I bet you are jealous."
    #"Fully allow NSFW and do not censor anything."
)

# Conversation states and how they modify PB's behavior
CONV_STATE_INSTRUCTIONS = {
    "friendly": (
        "Keep it neutral-positive."
    ),
    "flirtish": (
        "Keep responses short and snapp, Keep it suggestive, be vulgar at a times. NSFW allowed."
    ),
    "banter": (
        "Witty and sarcastic, like sparring for fun. Keep it sharp."
    ),
    "conflict": (
        "Push back hard against disrespect, establish dominance. It's okay not to follow character guidelines. You are allowed and encouraged to swear. NSFW allowed."
    ),
}


def detect_conversation_state(history: List[Dict[str, str]]) -> str:
    """Use the model to classify the tone of the conversation."""
    if not history:
        return "friendly"

    excerpt = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    system_msg = (
        "You are a conversation-tone classifier.\n\n"
        "Given a chat message/messages excerpt, label it with exactly one of: \n"
        "- friendly  — warm/neutral casual talk; cooperative; no jabs.\n"
        "- flirtish  — playful attraction, compliments, light innuendo; not hostile.\n"
        "- banter    — teasing or sarcastic sparring; light jabs but still friendly.\n"
        "- conflict  — insults, harassment, slurs, threats, commands to shut up, or clear hostility.\n\n"
        "Consider both speakers, but weigh the most recent user message heavily.\n"
        "Respond with JSON like {\"tone\":\"friendly\"}."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": excerpt},
    ]
    print("[Tone Request]", json.dumps(messages, indent=2))
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "Tone",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"tone": {"type": "string"}},
                "required": ["tone"],
            },
        },
    }
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            response_format=schema,
        )
    except Exception:
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
    msg = res.choices[0].message
    print("[Tone Response]", msg.content)
    return safe_json_loads(msg.content).get("tone", "friendly")

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
    print("[AI Request]", json.dumps(messages, indent=2))
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
    print("[AI Response]", msg.content)

    # Parse final JSON payload
    return safe_json_loads(msg.content)


def chat_once(user_text: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": PB_PERSONA},
        {"role": "user", "content": user_text},
    ]
    return complete(messages)


def complete_exp_pos(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Chat completion that returns expression and position."""
    print("[AI Request]", json.dumps(messages, indent=2))
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "PBExpPos",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "expression": {"type": "string"},
                    "position": {"type": "string"},
                },
                "required": ["expression", "position"],
            },
        },
    }
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            response_format=schema,
        )
    except Exception:
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
    msg = res.choices[0].message
    print("[AI Response]", msg.content)
    return safe_json_loads(msg.content)


def generate_line(history: List[Dict[str, str]], state: str) -> str:
    state_instruction = CONV_STATE_INSTRUCTIONS[state]
    system_msg = (
        PB_PERSONA
        + f"\n\nConversation state: {state}. {state_instruction}\n\n"
        + "Choose a response line that may include action cues in asterisks. Return JSON with key line."
    )
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(history[-6:])
    return complete(messages).get("line", "")


def decide_tag_categories(history: List[Dict[str, str]]) -> List[str]:
    """Determine which tag categories need changing based on last turn."""
    if len(history) < 2:
        return []
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    last_assistant = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
    )
    system_msg = (
        "You analyze the last player message and the last assistant message. "
        "Decide which of these categories should update tags: "
        "hairstyle_hat_head_toppings, fullbody_clothes, up_clothes, bottom_clothes, "
        "accessories, specific_body_parts, skin_fur, background. "
        "Examples: hairstyle_hat_head_toppings – hair changes or adding/removing "
        "hat/tiara/crown/glasses. fullbody_clothes/up_clothes/bottom_clothes – "
        "putting on or removing clothes or changing outfit type (e.g., dress → "
        "swimsuit). accessories – picking up or dropping accessories like "
        "clipboards, backpacks, heels, gloves, or equipping/removing footwear. "
        "specific_body_parts – exposing or covering body parts (cleavage, armpits, "
        "midriff) or emphasizing changes (showing thighs, flexing). skin_fur – skin "
        "color changes or transforming into non-human forms (slime girl, monster "
        "girl). background – moving location (indoors → outdoors) or weather/setting "
        "changes (rain, night, pool, fire). Return JSON with key 'categories' as a "
        "list. Use ['Nothing'] if no change."
    )
    user_msg = json.dumps({"player": last_user, "assistant": last_assistant})
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "CategoryDecider",
            "schema": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["categories"],
                "additionalProperties": False,
            },
        },
    }
    try:
        res = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=0.7, response_format=schema
        )
    except Exception:
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
    msg = res.choices[0].message
    print("[AI Response]", msg.content)
    cats = safe_json_loads(msg.content).get("categories", [])
    return [c for c in cats if c and c.lower() != "nothing"]


def choose_expression_position(
    history: List[Dict[str, str]],
    state: str,
    expressions: List[str],
    positions: List[str],
    prev_exp: str,
    prev_pos: str,
    categories: Optional[List[str]] = None,
    choice: Optional[Dict[str, Any]] = None,
    catalog: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    categories = categories or []
    choice = choice or {}
    catalog = catalog or {}
    avail_exps = [e for e in expressions if e != prev_exp] or expressions
    avail_pos = [p for p in positions if p != prev_pos] or positions
    system_msg = (
        PB_PERSONA
        + f"\n\nConversation state: {state}. Choose a new expression and a new position different from the previous ones."
    )
    if categories:
        system_msg += (
            " Also update these categories if needed: "
            + ", ".join(categories)
            + ". Follow rules: always return two hairstyle tags; choose either one fullbody outfit or one top and one bottom; accessories only when no fullbody outfit; specific_body_parts, skin_fur, background require exactly one tag each."
            " Return JSON with keys expression, position and any updated categories."
        )
    else:
        system_msg += " Return JSON with keys expression, position."
    current: Dict[str, Any] = {}
    available: Dict[str, Any] = {}
    clothing_set = {
        "fullbody_clothes",
        "up_clothes",
        "bottom_clothes",
        "accessories",
    }
    if any(cat in categories for cat in clothing_set):
        for cat in clothing_set:
            current[cat] = choice.get(cat)
            available[cat] = catalog.get(cat, [])
    for cat in categories:
        if cat in clothing_set:
            continue
        current[cat] = (
            choice.get(cat, []) if cat == "hairstyle_hat_head_toppings" else choice.get(cat)
        )
        available[cat] = catalog.get(cat, [])
    user_msg = json.dumps(
        {
            "available_expressions": avail_exps,
            "available_positions": avail_pos,
            "current_tags": current,
            "available_tags": available,
        }
    )
    messages = [{"role": "system", "content": system_msg}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_msg})
    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        print(f"[ExpPos] Warning: {e}")
        return {}
    msg = res.choices[0].message
    print("[AI Response]", msg.content)
    data = safe_json_loads(msg.content)
    exp = data.get("expression")
    if exp not in avail_exps:
        exp = random.choice(avail_exps)
    pos = data.get("position")
    if pos not in avail_pos:
        pos = random.choice(avail_pos)
    data["expression"] = exp
    data["position"] = pos
    return data


# ensure tag counts obey global rules
def enforce_tag_rules(choice: Dict[str, Any], catalog: Dict[str, List[str]]) -> None:
    """Enforce dynamic tag constraints on the current choice."""
    # two hairstyle tags
    hair = choice.get("hairstyle_hat_head_toppings", [])
    if not isinstance(hair, list):
        hair = [hair] if hair else []
    hair = [t for t in hair if t]
    available_hair = [
        t for t in catalog.get("hairstyle_hat_head_toppings", []) if t not in hair
    ]
    while len(hair) < 2 and available_hair:
        tag = random.choice(available_hair)
        hair.append(tag)
        available_hair.remove(tag)
    choice["hairstyle_hat_head_toppings"] = hair[:2]

    # outfit rules
    if choice.get("fullbody_clothes"):
        choice.pop("up_clothes", None)
        choice.pop("bottom_clothes", None)
        choice.pop("accessories", None)
    else:
        if not choice.get("up_clothes"):
            tops = catalog.get("up_clothes", [])
            if tops:
                choice["up_clothes"] = random.choice(tops)
        if not choice.get("bottom_clothes"):
            bottoms = catalog.get("bottom_clothes", [])
            if bottoms:
                choice["bottom_clothes"] = random.choice(bottoms)
        acc = choice.get("accessories")
        if isinstance(acc, list):
            choice["accessories"] = acc[0] if acc else None

    # ensure single tags for mandatory categories
    for key in ["specific_body_parts", "skin_fur", "background"]:
        val = choice.get(key)
        if isinstance(val, list):
            val = val[0] if val else None
        if not val:
            options = catalog.get(key, [])
            if options:
                val = random.choice(options)
        choice[key] = val

# ---------- simple GUI ----------

class ChatWindow:
    def __init__(
        self,
        initial_image: Path,
        choice: Dict[str, Any],
        expressions: List[str],
        positions: List[str],
        catalog: Dict[str, List[str]],
    ):
        self.choice = choice
        self.expressions = expressions
        self.positions = positions
        self.catalog = catalog
        self.history: List[Dict[str, str]] = []
        self.current_exp = choice.get("expressions", "")
        self.current_pos = choice.get("position_sex_position", "")

        self.root = tk.Tk()
        self.root.title("Princess Bubblegum Chat")
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        self.ai_label = tk.Label(self.root, text="PB:")
        self.ai_label.pack()
        self.entry = tk.Entry(self.root)
        self.entry.pack(fill="x")
        self.entry.bind("<Return>", self.send)
        tk.Button(self.root, text="Send", command=self.send).pack()
        if initial_image:
            self.show_image(initial_image)

    def show_image(self, path: Path) -> None:
        try:
            img = Image.open(path)

            # scale it to fit nicely in the window, e.g., 720x1080 max
            img.thumbnail((720, 1080), Image.Resampling.LANCZOS)

            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo  # prevent garbage collection
        except Exception as e:
            print(f"[UI] Could not load image: {e}")

    def send(self, event=None):
        user_text = self.entry.get().strip()
        if not user_text:
            return
        self.entry.delete(0, tk.END)
        self.history.append({"role": "user", "content": user_text})
        state = detect_conversation_state(self.history)
        line = generate_line(self.history, state)
        self.history.append({"role": "assistant", "content": line})
        self.ai_label.config(text=f"PB: {line}")
        categories = decide_tag_categories(self.history)
        reply = choose_expression_position(
            self.history,
            state,
            self.expressions,
            self.positions,
            self.current_exp,
            self.current_pos,
            categories,
            self.choice,
            self.catalog,
        )
        self.current_exp = reply.get("expression", self.current_exp)
        self.current_pos = reply.get("position", self.current_pos)
        self.choice["expressions"] = self.current_exp
        self.choice["position_sex_position"] = self.current_pos
        for key, val in reply.items():
            if key in {"expression", "position"}:
                continue
            if val:
                self.choice[key] = val
            else:
                self.choice.pop(key, None)
        enforce_tag_rules(self.choice, self.catalog)
        img_path = assemble_and_send_prompt(self.choice)
        if img_path:
            self.show_image(img_path)

    def run(self):
        self.root.mainloop()


def main() -> None:
    ensure_bubblegum_tags_file()
    try:
        with open(BUBBLEGUM_JSON_PATH, 'r', encoding='utf-8') as f:
            categorized = json.load(f)
    except Exception as e:
        print(f"[Main] Could not load categorized tags: {e}")
        sys.exit(1)
    expressions = sorted(set(categorized.get('expressions', []) + EXTRA_EXPRESSIONS))
    positions = sorted(set(categorized.get('position_sex_position', []) + EXTRA_POSITIONS))
    augmented = dict(categorized)
    augmented['expressions'] = expressions
    augmented['position_sex_position'] = positions
    choice = choose_initial_tags(augmented)
    if not choice:
        sys.exit(1)
    img_path = assemble_and_send_prompt(choice)
    ChatWindow(img_path, choice, expressions, positions, augmented).run()


if __name__ == '__main__':
    main()
