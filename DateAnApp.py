#!/usr/bin/env python3
"""Date an App - chat and ComfyUI image generator

This script lets a user pick an application, have AI describe it and turn it
into a character, then chat with that character while generating images via
ComfyUI.  It is a simplified variant of AIProject.py using a zero-LoRA
workflow.
"""
import os
import re
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import requests
from dotenv import load_dotenv
from openai import OpenAI

# ---------- environment & client ----------

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
BASE_URL = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
MODEL = os.getenv("TOGETHER_CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")

if not TOGETHER_API_KEY:
    raise SystemExit("Missing TOGETHER_API_KEY in .env")

client = OpenAI(api_key=TOGETHER_API_KEY, base_url=BASE_URL)

# ---------- paths & constants ----------

BASE_POSITIVE = "high quality"
BASE_NEGATIVE = (
    "NSFW, low quality, jpeg artifacts, deformed, extra fingers, text, watermark, logo, censored"
)

IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

COMFY_OUTPUT_DIR = Path(
    os.getenv("COMFY_OUTPUT_DIR", r"E:\\ConfyUI_New\\ComfyUI\\output")
)
WATCH_DIR = COMFY_OUTPUT_DIR / IMAGES_DIR.name
WATCH_DIR.mkdir(parents=True, exist_ok=True)

EXTRA_TAGS_PATH = os.path.join(os.path.dirname(__file__), "extra_tags.json")
try:
    with open(EXTRA_TAGS_PATH, "r", encoding="utf-8") as f:
        EXTRA_TAGS: Dict[str, List[str]] = json.load(f)
except Exception:
    EXTRA_TAGS = {}

# ---------- utilities ----------

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

def log_prompt(title: str, messages: List[Dict[str, str]]) -> None:
    """Print messages sent to the AI in a readable format."""
    print(f"\n[Prompt] {title}")
    for msg in messages:
        role = msg.get("role", "").capitalize()
        content = msg.get("content", "")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                content = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                pass
        print(f"{role}:\n{content}\n")

# ---------- conversation tone ----------

CONV_STATE_INSTRUCTIONS = {
    "friendly": (
        "Keep it neutral-positive."
    ),
    "flirtish": (
        "Keep responses short and snappy, Keep it suggestive at most."
    ),
    "banter": (
        "Witty and sarcastic, like sparring for fun. Keep it sharp."
    ),
    "conflict": (
        "Push back hard against disrespect, establish dominance. It's okay not to follow character guidelines. You are allowed and encouraged to swear."
    ),
}


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
    return {"tone": maybe_json.strip()}


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

# ---------- initial setup ----------

def resolve_shortcut(path: str) -> str:
    """Resolve a Windows .lnk shortcut to its target path."""
    if path.lower().endswith(".lnk"):
        try:
            import win32com.client  # type: ignore

            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(path)
            if shortcut.Targetpath:
                return shortcut.Targetpath
        except Exception:
            pass
    return path


def get_product_name(path: str) -> str:
    """Get product name from executable metadata or fallback names."""
    target = resolve_shortcut(path)
    name = Path(target).stem
    try:
        import win32api  # type: ignore

        info = win32api.GetFileVersionInfo(target, "\\")
        trans = win32api.VerQueryValue(info, r"\VarFileInfo\Translation")[0]
        lang_code = f"{trans[0]:04X}{trans[1]:04X}"
        product = win32api.VerQueryValue(
            info, rf"\StringFileInfo\{lang_code}\ProductName"
        )
        if product:
            return product
    except Exception:
        pass
    parent = os.path.basename(os.path.dirname(target))
    return parent or name


def _find_exe_in_dir(directory: Path) -> str | None:
    """Return path to a non-generic exe in directory (depth 1)."""
    if not directory.exists():
        return None
    generic = {
        "setup",
        "update",
        "uninstall",
        "launcher",
        "helper",
        "install",
    }
    for exe in directory.glob("*.exe"):
        name = get_product_name(str(exe))
        if name.lower() not in generic and exe.is_file():
            return str(exe)
    return None


def gather_registry_apps() -> List[Tuple[str, str]]:
    """Read Windows uninstall registry keys for installed apps."""
    apps: List[Tuple[str, str]] = []
    try:  # pragma: no cover - Windows only
        import winreg

        roots = [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]
        subkeys = [
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
            r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
        ]
        seen: set[str] = set()
        for root in roots:
            for sub in subkeys:
                try:
                    with winreg.OpenKey(root, sub) as hkey:
                        for i in range(winreg.QueryInfoKey(hkey)[0]):
                            try:
                                skn = winreg.EnumKey(hkey, i)
                                with winreg.OpenKey(hkey, skn) as sk:
                                    name = winreg.QueryValueEx(sk, "DisplayName")[0]
                                    icon = ""
                                    loc = ""
                                    try:
                                        icon = winreg.QueryValueEx(sk, "DisplayIcon")[0]
                                    except Exception:
                                        pass
                                    try:
                                        loc = winreg.QueryValueEx(sk, "InstallLocation")[0]
                                    except Exception:
                                        pass
                                    path = ""
                                    if icon:
                                        path = icon.split(",")[0].strip('"')
                                    elif loc:
                                        path = _find_exe_in_dir(Path(loc)) or ""
                                    if name and path and path.lower() not in seen:
                                        apps.append((name, path))
                                        seen.add(path.lower())
                            except Exception:
                                continue
                except Exception:
                    continue
    except Exception:
        return []
    return apps


def gather_shortcuts() -> List[Tuple[str, str]]:
    """Collect .lnk files from Desktop and Start Menu.

    Microsoft Store applications often use generic executables such as
    ``explorer.exe`` as the shortcut target.  When we deduplicate based solely
    on that target path, distinct apps (e.g. Spotify, Netflix) collapse into a
    single entry and thus fail to appear in the list.  Here we instead dedupe by
    the shortcut's name and fall back to that name if product metadata resolves
    to a generic system executable.
    """

    locations = []
    user = os.environ.get("USERPROFILE", "")
    public = os.environ.get("PUBLIC", "")
    appdata = os.environ.get("APPDATA", "")
    if user:
        locations.append(Path(user) / "Desktop")
    if public:
        locations.append(Path(public) / "Desktop")
    if appdata:
        locations.append(Path(appdata) / r"Microsoft\Windows\Start Menu\Programs")

    entries: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for loc in locations:
        if loc.exists():
            for p in loc.rglob("*.lnk"):
                target = resolve_shortcut(str(p))
                name = get_product_name(str(p))
                # Many store apps resolve to explorer.exe/cmd.exe; use the
                # shortcut file name instead in such cases.
                if not name or Path(target).name.lower() in {"explorer.exe", "cmd.exe"}:
                    name = p.stem
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                entries.append((name, target))
    return sorted(entries, key=lambda x: x[0].lower())


def gather_steam_games() -> List[Tuple[str, str]]:
    """Return installed Steam games using library manifests."""
    games: List[Tuple[str, str]] = []
    try:  # pragma: no cover - Windows only
        import winreg

        steam_path = ""
        for root in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
            try:
                with winreg.OpenKey(root, r"SOFTWARE\Valve\Steam") as k:
                    steam_path = winreg.QueryValueEx(k, "SteamPath")[0]
                    break
            except Exception:
                continue
        if not steam_path:
            return []
        steam_path = steam_path.replace("/", "\\")
        libraries = [Path(steam_path) / "steamapps"]
        lib_file = libraries[0] / "libraryfolders.vdf"
        if lib_file.exists():
            txt = lib_file.read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(r'"\d+"\s*"([^"]+)"', txt):
                libraries.append(Path(m.group(1).replace("/", "\\")) / "steamapps")
        for lib in libraries:
            for manifest in lib.glob("appmanifest_*.acf"):
                try:
                    content = manifest.read_text(encoding="utf-8", errors="ignore")
                    name_m = re.search(r'"name"\s*"([^"]+)"', content)
                    dir_m = re.search(r'"installdir"\s*"([^"]+)"', content)
                    if not name_m or not dir_m:
                        continue
                    name = name_m.group(1)
                    installdir = lib / "common" / dir_m.group(1)
                    path = _find_exe_in_dir(installdir)
                    if path:
                        games.append((name, path))
                except Exception:
                    continue
    except Exception:
        return []
    return games


def gather_riot_games() -> List[Tuple[str, str]]:
    """Enumerate Riot-installed games."""
    games: List[Tuple[str, str]] = []
    base = Path(r"C:\Riot Games")
    if base.exists():
        for child in base.iterdir():
            if child.is_dir():
                exe = _find_exe_in_dir(child)
                if exe:
                    name = get_product_name(exe)
                    games.append((name, exe))
    meta_dir = Path(r"C:\ProgramData\Riot Games\Metadata")
    if meta_dir.exists():
        for meta in meta_dir.glob("*.json"):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                name = data.get("product_name") or data.get("name")
                path = data.get("product_install_full_path")
                if name and path:
                    games.append((name, path))
            except Exception:
                continue
    return games


def scan_common_folders() -> List[Tuple[str, str]]:
    """Fallback: scan common install folders for executables."""
    dirs: List[Path] = []
    for env in ["ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"]:
        p = os.environ.get(env)
        if p:
            dirs.append(Path(p))
    for extra in [Path(r"C:\Games"), Path(r"D:\Games")]:
        dirs.append(extra)
    results: List[Tuple[str, str]] = []
    seen: set[str] = set()
    generic = {
        "setup",
        "update",
        "launcher",
        "install",
        "uninstall",
    }
    for base in dirs:
        if not base.exists():
            continue
        for root, subdirs, files in os.walk(base):
            depth = Path(root).relative_to(base).parts
            if len(depth) >= 3:
                subdirs[:] = []
            for f in files:
                if not f.lower().endswith(".exe"):
                    continue
                path = os.path.join(root, f)
                name = get_product_name(path)
                if name.lower() in generic:
                    continue
                key = path.lower()
                if key not in seen:
                    results.append((name, path))
                    seen.add(key)
    return results


def gather_apps() -> List[Tuple[str, str]]:
    """Aggregate installed applications from various providers."""
    providers = [
        gather_registry_apps,
        gather_shortcuts,
        gather_steam_games,
        gather_riot_games,
        scan_common_folders,
    ]
    # Deduplicate by application name rather than path.  Some providers (notably
    # Microsoft Store shortcuts) may point multiple apps to the same executable
    # such as ``explorer.exe``.  Using the path as the key would therefore drop
    # many distinct entries.
    apps: dict[str, Tuple[str, str]] = {}
    for prov in providers:
        try:
            for name, path in prov():
                key = name.lower()
                if key not in apps:
                    apps[key] = (name, path)
        except Exception:
            continue
    return sorted(apps.values(), key=lambda x: x[0].lower())


def choose_app() -> Tuple[str, str]:
    """Show a list of detected apps and return chosen name and path."""
    apps = gather_apps()
    if apps:
        root = tk.Tk()
        root.title("Choose Application")
        lb = tk.Listbox(root, width=50, height=20)
        for name, _ in apps:
            lb.insert(tk.END, name)
        lb.pack()
        result: Dict[str, str] = {}

        def on_select(event=None):
            sel = lb.curselection()
            if sel:
                result["name"] = apps[sel[0]][0]
                result["path"] = apps[sel[0]][1]
                root.destroy()

        lb.bind("<Double-Button-1>", on_select)
        tk.Button(root, text="OK", command=on_select).pack()
        root.mainloop()
        if "name" in result:
            print(f"[Main] Selected: {result['name']}")
            return result["name"], result["path"]
        raise SystemExit("No app chosen")

    # Fallback to manual browsing
    print("[Main] Please choose an application file.")
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()
    root.destroy()
    if not path:
        raise SystemExit("No file chosen")
    name = get_product_name(path)
    print(f"[Main] Selected file: {path}")
    return name, path

def recognize_app(app_name: str) -> str:
    print(f"[AI] Recognizing app '{app_name}'...")
    system_msg = "You are an AI app recognizer program."
    user_msg = (
        f"Program: {app_name}\n"
        "1. What is this program/game/app?\n"
        "2. What do internet users and memes think about this app?\n"
        "3. What this app icon looks like and what color pallete it uses (what are the leading colors)?"
    )
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    log_prompt("App recognition", messages)
    res = client.chat.completions.create(model=MODEL, messages=messages)
    content = res.choices[0].message.content.strip()
    print(f"[AI] Recognition result:\n{content}")
    return content

def create_character(app_name: str, app_info: str) -> Dict[str, str]:
    print("[AI] Creating character description...")
    system_msg = "You design characters based on apps. Respond in JSON."
    user_msg = (
        f"App name: {app_name}\nInfo: {app_info}\n"
        "Based on this app/program/game name, it's icon, it's description and memes answer following questions:\n"
        "1. What would this icon look like as a character?\n"
        "2. What would this icon be called as a character? (Perhaps make a simple pun based on the icon name)\n"
        "3. What kind of character could this app have (consider internet users' opinions)\n\n"
        "Return JSON with keys name, sex (male/female), appearance (3 sentences), personality (3 sentences)."
    )
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    log_prompt("Character creation", messages)
    res = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=messages,
        response_format={"type": "json_object"},
    )
    data = json.loads(res.choices[0].message.content)
    print(f"[AI] Character data:\n{json.dumps(data, indent=2)}")
    return {
        "name": data.get("name", "Unknown"),
        "sex": data.get("sex", "female"),
        "appearance": data.get("appearance", ""),
        "personality": data.get("personality", ""),
    }

def generate_tags(character: Dict[str, str]) -> Dict[str, str]:
    print("[AI] Generating character tags...")
    system_msg = (
        "Create simple two-word tags for a character. Respond in JSON with keys:"
        " hair_type, hair_color, body_type, skin, accessories, background, clothes1, clothes2."
    )
    user_msg = (
        f"Character name: {character['name']}\n"
        f"Sex: {character['sex']}\n"
        f"Appearance: {character['appearance']}\n"
        f"Personality: {character['personality']}"
    )
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
    log_prompt("Tag generation", messages)
    res = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=messages,
        response_format={"type": "json_object"},
    )
    tags = json.loads(res.choices[0].message.content)
    print(f"[AI] Tags:\n{json.dumps(tags, indent=2)}")
    return tags

def choose_expression_position(
    prev_exp: str, prev_pos: str, expressions: List[str], positions: List[str]
) -> Tuple[str, str]:
    print("[AI] Choosing new expression and position...")
    avail_exp = [e for e in expressions if e != prev_exp] or expressions
    avail_pos = [p for p in positions if p != prev_pos] or positions
    system_msg = (
        "Choose a new expression and a new position from the provided lists. "
        "They must be different from the previous ones. Return JSON with keys expression and position."
    )
    payload = {
        "expressions": avail_exp,
        "positions": avail_pos,
        "previous_expression": prev_exp,
        "previous_position": prev_pos,
    }
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(payload)},
    ]
    log_prompt("Expression/position choice", messages)
    res = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=messages,
        response_format={"type": "json_object"},
    )
    data = json.loads(res.choices[0].message.content)
    print(f"[AI] Chosen expression: {data.get('expression')}, position: {data.get('position')}")
    return data.get("expression", prev_exp), data.get("position", prev_pos)

# ---------- ComfyUI prompt ----------

def assemble_and_send_prompt(
    sex: str, tags: Dict[str, str], token: str, app_name: str
) -> Path | None:
    print("[ComfyUI] Assembling prompt...")
    parts: List[str] = []
    if tags.get("hair_type"):
        parts.append(f"{tags['hair_type']} hair")
    if tags.get("hair_color"):
        parts.append(f"{tags['hair_color']} hair")
    if tags.get("body_type"):
        parts.append(tags["body_type"])
    if tags.get("expression"):
        parts.append(tags["expression"])
    if tags.get("position"):
        parts.append(tags["position"])
    if tags.get("skin"):
        parts.append(tags["skin"])
    if tags.get("accessories"):
        parts.append(tags["accessories"])
    if tags.get("background"):
        parts.append(tags["background"] + " background")
    if tags.get("clothes1"):
        parts.append(tags["clothes1"])
    if tags.get("clothes2"):
        parts.append(tags["clothes2"])
    if app_name:
        parts.append(f"clothes with {app_name} logo")
    positive_text = BASE_POSITIVE + f", 1{sex}" + (", " + ", ".join(parts) if parts else "")

    try:
        with open("0lora.json", "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        print(f"[Prompt] Could not load workflow template: {e}")
        return None

    print(f"[ComfyUI] Positive prompt: {positive_text}")
    print(f"[ComfyUI] Negative prompt: {BASE_NEGATIVE}")

    workflow.get("6", {}).setdefault("inputs", {})["text"] = positive_text
    workflow.get("7", {}).setdefault("inputs", {})["text"] = BASE_NEGATIVE
    workflow.get("3", {}).setdefault("inputs", {})["seed"] = 504
    workflow.get("4", {}).setdefault("inputs", {})["ckpt_name"] = "waiNSFWIllustrious_v120.safetensors"
    workflow.get("12", {}).setdefault("inputs", {})["model_name"] = "RealESRGAN_x4plus.pth"

    short_slug = "__".join(sanitize(t) for t in parts[:3])
    prefix_base = f"{sanitize(token)}__{short_slug}" if short_slug else sanitize(token)
    n = next_run_number(IMAGES_DIR, prefix_base)
    filename_prefix = f"{IMAGES_DIR.name}/{prefix_base}__run{n}"
    workflow.get("9", {}).setdefault("inputs", {})["filename_prefix"] = filename_prefix
    expected_prefix = Path(filename_prefix).name

    existing = set(WATCH_DIR.glob(f"{expected_prefix}*.png"))

    payload = {"prompt": workflow}
    try:
        r = requests.post("http://127.0.0.1:8188/prompt", json=payload, timeout=10)
        print(f"[ComfyUI] POST status {r.status_code}")
        print(f"[ComfyUI] Response body: {r.text}")
    except Exception as e:
        print(f"[ComfyUI] Warning: {e}")
        return None

    for _ in range(240):
        new_files = set(WATCH_DIR.glob(f"{expected_prefix}*.png")) - existing
        if new_files:
            candidate = max(new_files, key=os.path.getmtime)
            last_size = -1
            for _ in range(20):
                try:
                    sz = candidate.stat().st_size
                    if sz == last_size:
                        with Image.open(candidate) as im:
                            im.verify()
                        return candidate
                    last_size = sz
                    time.sleep(0.5)
                except Exception:
                    time.sleep(0.5)
        time.sleep(0.5)
    print("[ComfyUI] Warning: image not found or incomplete")
    return None

# ---------- chat window ----------

def recent_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    user_idx: List[int] = []
    ai_idx: List[int] = []
    for i in range(len(history) - 1, -1, -1):
        role = history[i]["role"]
        if role == "user" and len(user_idx) < 3:
            user_idx.append(i)
        elif role == "assistant" and len(ai_idx) < 3:
            ai_idx.append(i)
        if len(user_idx) >= 3 and len(ai_idx) >= 3:
            break
    idx = sorted(user_idx + ai_idx)
    return [history[i] for i in idx]

class ChatWindow:
    def __init__(self, app_name: str, character: Dict[str, str], tags: Dict[str, str]):
        self.app_name = app_name
        self.character = character
        self.base_tags = tags
        self.expressions = EXTRA_TAGS.get("expressions", [])
        self.positions = EXTRA_TAGS.get("position_sex_position", [])
        self.expression = ""
        self.position = ""
        self.history: List[Dict[str, str]] = []
        self.token = sanitize(character["name"] or app_name)
        self.base_prompt = (
            f"You are {character['name']}, a character based on {app_name}. "
            f"Appearance: {character['appearance']} "
            f"Personality: {character['personality']} "
            f"Background: {tags.get('background', '')}. Stay in character."
        )

        self.root = tk.Tk()
        self.root.title(character["name"])
        self.image_label = tk.Label(self.root)
        self.image_label.pack()
        self.ai_label = tk.Label(self.root, text=f"{character['name']}:")
        self.ai_label.pack()
        self.entry = tk.Entry(self.root)
        self.entry.pack(fill="x")
        self.entry.bind("<Return>", self.send)
        tk.Button(self.root, text="Send", command=self.send).pack()
        self.initial_message()

    def initial_message(self) -> None:
        print("[Chat] Sending initial message to AI...")
        state_instruction = CONV_STATE_INSTRUCTIONS["friendly"]
        system_prompt = self.base_prompt + f" Conversation state: friendly. {state_instruction}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Introduce yourself."},
        ]
        log_prompt("Initial chat", messages)
        res = client.chat.completions.create(model=MODEL, messages=messages)
        line = res.choices[0].message.content.strip()
        print(f"[AI] Initial reply:\n{line}")
        self.history.append({"role": "assistant", "content": line})
        self.ai_label.config(text=f"{self.character['name']}: {line}")
        self.update_image()

    def show_image(self, path: Path) -> None:
        try:
            img = Image.open(path)
            img.thumbnail((720, 1080), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo)
            self.image_label.image = self.photo
        except Exception as e:
            print(f"[UI] Could not load image: {e}")

    def update_image(self) -> None:
        exp, pos = choose_expression_position(
            self.expression, self.position, self.expressions, self.positions
        )
        self.expression = exp or self.expression
        self.position = pos or self.position
        tags = dict(self.base_tags)
        tags["expression"] = self.expression
        tags["position"] = self.position
        img_path = assemble_and_send_prompt(
            self.character["sex"], tags, self.token, self.app_name
        )
        if img_path:
            self.show_image(img_path)

    def send(self, event=None):
        user_text = self.entry.get().strip()
        if not user_text:
            return
        self.entry.delete(0, tk.END)
        self.history.append({"role": "user", "content": user_text})
        state = detect_conversation_state(self.history)
        instruction = CONV_STATE_INSTRUCTIONS.get(state, "")
        system_prompt = self.base_prompt + f" Conversation state: {state}. {instruction}"
        msgs = [{"role": "system", "content": system_prompt}]
        msgs.extend(recent_messages(self.history))
        log_prompt("Chat", msgs)
        res = client.chat.completions.create(model=MODEL, messages=msgs)
        reply = res.choices[0].message.content.strip()
        print(f"[AI] Reply:\n{reply}")
        self.history.append({"role": "assistant", "content": reply})
        self.ai_label.config(text=f"{self.character['name']}: {reply}")
        self.update_image()

    def run(self) -> None:
        self.root.mainloop()

# ---------- main ----------

def main() -> None:
    print("[Main] Starting Date an App...")
    app_name, app_path = choose_app()
    app_info = recognize_app(app_name)
    character = create_character(app_name, app_info)
    tags = generate_tags(character)
    print("[Main] Launching chat window.")
    ChatWindow(app_name, character, tags).run()

if __name__ == "__main__":
    main()
