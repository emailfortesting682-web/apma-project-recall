# modules/embeddings_engine.py

import os
import json
from pathlib import Path
from openai import OpenAI
import streamlit as st
from . import supabase_store

# =====================================================
# API KEY HANDLING
# =====================================================
def get_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except:
        return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = get_api_key()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# =====================================================
# EMBEDDINGS ENGINE
# =====================================================
class EmbeddingsEngine:
    """
    Dynamic embeddings engine:
    - Embeds ALL meaningful user data
    - Works with manual entry & file upload
    - No hardcoded column dependency
    """

    # Columns that should NEVER be embedded
    EXCLUDED_COLS = {
        "AddedBy",
        "__index__"
    }

    def __init__(self):
        if client is None:
            raise ValueError(
                "OPENAI_API_KEY is not set. Configure it in Streamlit secrets or environment variables."
            )
        self.client = client

    # -------------------------------------------------
    def _text_for_row(self, row) -> str:
        """
        Build semantic text from ALL non-empty user columns.
        """
        parts = []

        for col, val in row.items():
            if col in self.EXCLUDED_COLS:
                continue

            val = str(val).strip()
            if not val or val.lower() == "nan":
                continue

            parts.append(f"{col}: {val}")

        return " | ".join(parts)

    # -------------------------------------------------
    def embed_texts(self, texts):
        """
        Generate embeddings for list of texts.
        """
        resp = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return [item.embedding for item in resp.data]

    # -------------------------------------------------
    def index_dataframe(self, memory_path, df, id_prefix="mem"):
        """
        Create and save embeddings for a dataframe.
        """

        texts = []
        row_ids = []

        for idx, row in df.iterrows():
            text = self._text_for_row(row)
            if text:
                texts.append(text)
                row_ids.append(int(idx))

        if not texts:
            raise ValueError("No semantic content found to embed.")

        embeddings = self.embed_texts(texts)

        mem_path = Path(memory_path)
        mem_id = mem_path.stem
        out_path = mem_path.parent / f"{mem_id}_embeddings.json"

        payload = {
            "model": "text-embedding-3-large",
            "row_ids": row_ids,
            "embeddings": embeddings
        }

        if supabase_store.is_enabled():
            supabase_store.save_embeddings(mem_id, payload)

        out_path.write_text(json.dumps(payload, ensure_ascii=False))
        return str(out_path)
