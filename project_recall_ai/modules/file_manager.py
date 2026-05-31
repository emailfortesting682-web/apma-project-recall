import os
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from .utils import safe_filename
from . import supabase_store

class MemoryManager:
    def __init__(self, data_dir='data'):
        self.base = Path(data_dir)
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
        if self.use_supabase:
            return supabase_store.list_memories()
        return list(self.index.keys())

    def list_memories_full(self):
        if self.use_supabase:
            return supabase_store.list_memories_full()
        return {k:v['memory_path'] for k,v in self.index.items()}

    def save_upload(self, uploaded_file):
        # uploaded_file is a streamlit UploadedFile
        fname = safe_filename(uploaded_file.name)
        dest = self.upload_dir / fname
        with open(dest, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return str(dest)

    def create_or_update_memory(self, name, df: pd.DataFrame, mode='Create new memory file', target_memory=None):
        mid = safe_filename(name)
        mem_path = str(self.mem_dir / f"{mid}.parquet")
        meta = {
            'memory_id': mid,
            'memory_path': mem_path,
            'timestamp': datetime.utcnow().isoformat(),
            'n_records': len(df)
        }
        if self.use_supabase:
            supabase_store.upsert_memory(mid, name, df, meta)
            return meta

        # write parquet
        df.to_parquet(mem_path, index=False)
        # update index
        self.index[mid] = meta
        self._save_index()
        return meta

    def load_memory_dataframe(self, mem_id):
        if self.use_supabase:
            return supabase_store.load_memory_dataframe(mem_id)
        meta = self.index.get(mem_id)
        if not meta:
            raise FileNotFoundError('Memory not found')
        return pd.read_parquet(meta['memory_path'])
