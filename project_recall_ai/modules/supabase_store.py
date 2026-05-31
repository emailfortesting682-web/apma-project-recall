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


def list_memories() -> list[str]:
    client = _client()
    if not client:
        return []
    res = client.table(MEMORIES_TABLE).select("memory_id").order("updated_at", desc=True).execute()
    return [row["memory_id"] for row in (res.data or [])]


def list_memories_full() -> dict[str, str]:
    client = _client()
    if not client:
        return {}
    res = client.table(MEMORIES_TABLE).select("memory_id,name").order("updated_at", desc=True).execute()
    return {row["memory_id"]: row.get("name") or row["memory_id"] for row in (res.data or [])}


def upsert_memory(memory_id: str, name: str, df: pd.DataFrame, meta: dict[str, Any]) -> None:
    client = _client()
    if not client:
        return
    records = df.fillna("").astype(str).to_dict(orient="records")
    client.table(MEMORIES_TABLE).upsert(
        {
            "memory_id": memory_id,
            "name": name,
            "records": records,
            "metadata": meta,
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
