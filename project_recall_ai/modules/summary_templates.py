import json
from pathlib import Path
from . import supabase_store

TEMPLATE_FILE = Path("data/summary_templates.json")
TEMPLATE_KEY = "summary_templates"

DEFAULT_TEMPLATE = {
    "Default": {
        "sections": [
            "Context",
            "Problem",
            "Solution",
            "Lessons Learned"
        ],
        "tone": "simple",
        "length": "short"
    }
}

def load_templates():
    if supabase_store.is_enabled():
        local_default = json.loads(TEMPLATE_FILE.read_text()) if TEMPLATE_FILE.exists() else DEFAULT_TEMPLATE
        data = supabase_store.load_kv(TEMPLATE_KEY, local_default)
        if not data:
            data = DEFAULT_TEMPLATE
        if data == local_default:
            supabase_store.save_kv(TEMPLATE_KEY, data)
        return data

    if not TEMPLATE_FILE.exists():
        save_templates(DEFAULT_TEMPLATE)
        return DEFAULT_TEMPLATE

    try:
        data = json.loads(TEMPLATE_FILE.read_text())
        if not data:
            save_templates(DEFAULT_TEMPLATE)
            return DEFAULT_TEMPLATE
        return data
    except Exception:
        save_templates(DEFAULT_TEMPLATE)
        return DEFAULT_TEMPLATE

def save_templates(templates: dict):
    if supabase_store.is_enabled():
        supabase_store.save_kv(TEMPLATE_KEY, templates)
        return
    TEMPLATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEMPLATE_FILE.write_text(json.dumps(templates, indent=2))
