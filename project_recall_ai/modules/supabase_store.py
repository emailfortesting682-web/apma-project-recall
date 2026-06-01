import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

try:
    from supabase import create_client
except Exception:
    create_client = None


MEMORIES_TABLE = "apma_memories"
KV_TABLE = "apma_kv"


def _secret(name: str) -> str | None:
    try:
        return st.secrets.get(name) or os.getenv(name)
    except Exception:
        return os.getenv(name)


def is_enabled() -> bool:
    return bool(_secret("SUPABASE_URL") and _secret("SUPABASE_SERVICE_ROLE_KEY") and create_client)


def _client():
    if not is_enabled():
        return None
    return create_client(_secret("SUPABASE_URL"), _secret("SUPABASE_SERVICE_ROLE_KEY"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _can_access(metadata: dict[str, Any] | None, user_id: str | None) -> bool:
    if not user_id:
        return False
    metadata = metadata or {}
    owner_id = metadata.get("owner_id")
    allowed = metadata.get("allowed_user_ids") or ["*"]
    return "*" in allowed or user_id == owner_id or user_id in allowed


def list_memories(user_id: str | None = None) -> list[str]:
    client = _client()
    if not client:
        return []
    res = client.table(MEMORIES_TABLE).select("memory_id,metadata").order("updated_at", desc=True).execute()
    rows = res.data or []
    if user_id:
        rows = [row for row in rows if _can_access(row.get("metadata"), user_id)]
    return [row["memory_id"] for row in rows]


def list_memories_full(user_id: str | None = None) -> dict[str, str]:
    client = _client()
    if not client:
        return {}
    res = client.table(MEMORIES_TABLE).select("memory_id,name,metadata").order("updated_at", desc=True).execute()
    rows = res.data or []
    if user_id:
        rows = [row for row in rows if _can_access(row.get("metadata"), user_id)]
    return {row["memory_id"]: row.get("name") or row["memory_id"] for row in rows}


def load_memory_metadata(memory_id: str) -> dict[str, Any]:
    client = _client()
    if not client:
        return {}
    res = client.table(MEMORIES_TABLE).select("metadata").eq("memory_id", memory_id).limit(1).execute()
    if not res.data:
        return {}
    return res.data[0].get("metadata") or {}


def upsert_memory(memory_id: str, name: str, df: pd.DataFrame, meta: dict[str, Any]) -> None:
    client = _client()
    if not client:
        return
    existing_meta = load_memory_metadata(memory_id)
    merged_meta = {**existing_meta, **meta}
    audit_log = list(existing_meta.get("audit_log", []))
    if meta.get("audit_event"):
        audit_log.append(meta["audit_event"])
    merged_meta["audit_log"] = audit_log[-250:]

    records = df.fillna("").astype(str).to_dict(orient="records")
    client.table(MEMORIES_TABLE).upsert(
        {
            "memory_id": memory_id,
            "name": name,
            "records": records,
            "metadata": merged_meta,
            "updated_at": _now(),
        },
        on_conflict="memory_id",
    ).execute()


def load_memory_dataframe(memory_id: str) -> pd.DataFrame:
    client = _client()
    if not client:
        raise FileNotFoundError("Supabase is not configured")
    res = client.table(MEMORIES_TABLE).select("records").eq("memory_id", memory_id).limit(1).execute()
    if not res.data:
        raise FileNotFoundError("Memory not found")
    return pd.DataFrame(res.data[0].get("records") or [])


def save_embeddings(memory_id: str, payload: dict[str, Any]) -> None:
    client = _client()
    if not client:
        return
    client.table(MEMORIES_TABLE).update(
        {
            "embeddings": payload,
            "updated_at": _now(),
        }
    ).eq("memory_id", memory_id).execute()


def load_embeddings(memory_id: str) -> dict[str, Any] | None:
    client = _client()
    if not client:
        return None
    res = client.table(MEMORIES_TABLE).select("embeddings").eq("memory_id", memory_id).limit(1).execute()
    if not res.data:
        return None
    return res.data[0].get("embeddings")


def load_kv(key: str, default: Any) -> Any:
    client = _client()
    if not client:
        return default
    res = client.table(KV_TABLE).select("value").eq("key", key).limit(1).execute()
    if not res.data:
        return default
    return res.data[0].get("value", default)


def save_kv(key: str, value: Any) -> None:
    client = _client()
    if not client:
        return
    client.table(KV_TABLE).upsert(
        {
            "key": key,
            "value": value,
            "updated_at": _now(),
        },
        on_conflict="key",
    ).execute()
