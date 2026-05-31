# modules/user_manager.py
import json
from pathlib import Path
import hashlib
import os
from . import supabase_store

USERS_PATH = Path("data") / "users.json"
USERS_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_users():
    if supabase_store.is_enabled():
        return supabase_store.load_kv("users", {})
    if USERS_PATH.exists():
        try:
            return json.loads(USERS_PATH.read_text())
        except Exception:
            return {}
    return {}

def _save_users(users):
    if supabase_store.is_enabled():
        supabase_store.save_kv("users", users)
        return
    USERS_PATH.write_text(json.dumps(users, indent=2))

def _hash_password(id_number: str, password: str) -> str:
    """
    Derive a secure-ish password hash using PBKDF2 with the id_number as salt.
    Returns hex string.
    """
    salt = id_number.encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return dk.hex()

def create_user(first_name: str, last_name: str, id_number: str, password: str) -> (bool, str):
    """
    Create a new user. Returns (success, message).
    Enforces one account per id_number.
    """
    id_number = str(id_number).strip()
    if not id_number:
        return False, "ID number required."
    users = _load_users()
    if id_number in users:
        return False, "An account with this ID already exists."
    if not password or len(password) < 6:
        return False, "Password required (min 6 characters)."

    users[id_number] = {
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "password_hash": _hash_password(id_number, password)
    }
    _save_users(users)
    return True, "Account created."

def authenticate(id_number: str, password: str) -> (bool, str, dict):
    """
    Authenticate a user. Returns (success, message, user_dict_or_None)
    """
    id_number = str(id_number).strip()
    users = _load_users()
    u = users.get(id_number)
    if not u:
        return False, "No account found for this ID.", None
    expected = u.get("password_hash", "")
    if not expected:
        return False, "Account has no password set.", None
    if _hash_password(id_number, password) == expected:
        # return a reduced user profile
        return True, "Authenticated.", {
            "id": id_number,
            "first_name": u.get("first_name", ""),
            "last_name": u.get("last_name", "")
        }
    return False, "Invalid password.", None
