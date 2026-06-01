import os
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from .utils import safe_filename
from . import supabase_store

class MemoryManager:
    def __init__(self, data_dir='data', current_user=None):
        self.base = Path(data_dir)
        self.current_user = current_user or {}
        self.upload_dir = self.base / 'uploads'
        self.mem_dir = self.base / 'memories'
        self.index_fname = self.mem_dir / 'memories_index.json'
        self.use_supabase = supabase_store.is_enabled()
        self.index = self._load_index()

    def _load_index(self):
        if self.use_supabase:
            return {}
        if self.index_fname.exists():
            return json.loads(self.index_fname.read_text())
        return {}

    def _save_index(self):
        if self.use_supabase:
            return
        self.index_fname.write_text(json.dumps(self.index, indent=2))

    def list_memories(self):
        user_id = self.current_user.get("id")
        if self.use_supabase:
            return supabase_store.list_memories(user_id=user_id)
        memories = []
        for key, meta in self.index.items():
            allowed = meta.get("allowed_user_ids", ["*"])
            if not user_id or "*" in allowed or user_id == meta.get("owner_id") or user_id in allowed:
                memories.append(key)
        return memories

    def list_memories_full(self):
        user_id = self.current_user.get("id")
        if self.use_supabase:
            return supabase_store.list_memories_full(user_id=user_id)
        return {k:v['memory_path'] for k,v in self.index.items()}

    def save_upload(self, uploaded_file):
        # uploaded_file is a streamlit UploadedFile
        fname = safe_filename(uploaded_file.name)
        dest = self.upload_dir / fname
        with open(dest, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return str(dest)

    def create_or_update_memory(
        self,
        name,
        df: pd.DataFrame,
        mode='Create new memory file',
        target_memory=None,
        allowed_user_ids=None,
        schema=None,
        audit_action="save_memory",
    ):
        mid = safe_filename(name)
        mem_path = str(self.mem_dir / f"{mid}.parquet")
        existing_meta = self.get_memory_metadata(mid)
        user_id = self.current_user.get("id", "")
        username = " ".join(
            part for part in [
                self.current_user.get("first_name", ""),
                self.current_user.get("last_name", "")
            ] if part
        ).strip()
        allowed = allowed_user_ids or existing_meta.get("allowed_user_ids") or ["*"]
        resolved_schema = schema or {
            col: {"type": str(df[col].dtype)}
            for col in df.columns
        }
        meta = {
            'memory_id': mid,
            'memory_path': mem_path,
            'timestamp': datetime.utcnow().isoformat(),
            'n_records': len(df),
            'owner_id': existing_meta.get('owner_id') or user_id,
            'owner_name': existing_meta.get('owner_name') or username,
            'allowed_user_ids': allowed,
            'schema': resolved_schema,
            'updated_by': user_id,
            'updated_by_name': username,
            'updated_at': datetime.utcnow().isoformat(),
            'audit_event': {
                'action': audit_action,
                'user_id': user_id,
                'username': username,
                'timestamp': datetime.utcnow().isoformat(),
                'records': len(df),
            }
        }
        if self.use_supabase:
            supabase_store.upsert_memory(mid, name, df, meta)
            return meta

        # write parquet
        df.to_parquet(mem_path, index=False)
        # update index
        audit_log = list(existing_meta.get("audit_log", []))
        audit_log.append(meta["audit_event"])
        meta["audit_log"] = audit_log[-250:]
        self.index[mid] = {**existing_meta, **meta}
        self._save_index()
        return meta

    def load_memory_dataframe(self, mem_id):
        if self.use_supabase:
            return supabase_store.load_memory_dataframe(mem_id)
        meta = self.index.get(mem_id)
        if not meta:
            raise FileNotFoundError('Memory not found')
        return pd.read_parquet(meta['memory_path'])

    def get_memory_metadata(self, mem_id):
        if self.use_supabase:
            return supabase_store.load_memory_metadata(mem_id)
        return self.index.get(mem_id, {})

    def update_memory_permissions(self, mem_id, allowed_user_ids):
        meta = self.get_memory_metadata(mem_id)
        if not meta:
            return
        df = self.load_memory_dataframe(mem_id)
        self.create_or_update_memory(
            mem_id,
            df,
            allowed_user_ids=allowed_user_ids,
            schema=meta.get("schema"),
            audit_action="update_permissions",
        )
