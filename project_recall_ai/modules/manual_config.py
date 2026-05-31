import json
from pathlib import Path
from . import supabase_store

CONFIG_PATH = Path("data/manual_entry_config.json")
CONFIG_KEY = "manual_entry_config"

def load_config():
    if supabase_store.is_enabled():
        local_default = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
        cfg = supabase_store.load_kv(CONFIG_KEY, local_default)
        if cfg == local_default and local_default:
            supabase_store.save_kv(CONFIG_KEY, local_default)
        return cfg
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}

def save_config(cfg: dict):
    if supabase_store.is_enabled():
        supabase_store.save_kv(CONFIG_KEY, cfg)
        return
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
