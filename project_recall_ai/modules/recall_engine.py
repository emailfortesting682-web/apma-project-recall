# modules/recall_engine.py

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from textblob import TextBlob
from rapidfuzz import fuzz
from openai import OpenAI
import streamlit as st
from . import supabase_store

# =====================================================
# API KEY HANDLING
# =====================================================
def get_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return os.getenv("OPENAI_API_KEY")

_OPENAI_KEY = get_api_key()
_openai_client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None


# =====================================================
# UTILS
# =====================================================
def _cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if a.size == 0 or b.size == 0:
        return 0.0

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0

    return float(np.dot(a, b) / (na * nb))


def _row_text(row, columns=None):
    cols = columns or list(row.index)
    parts = []
    for col in cols:
        if col not in row.index:
            continue
        val = str(row[col]).strip()
        if val and val.lower() != "nan":
            parts.append(f"{col}: {val}")
    return " | ".join(parts)


# =====================================================
# RECALL ENGINE
# =====================================================
class RecallEngine:
    def __init__(
        self,
        emb_engine,
        mem_manager,
        category_col,
        phase_col,
        problem_col,
        solution_col,
        phase_match_threshold=75,
        category_match_threshold=75,
    ):
        self.emb_engine = emb_engine
        self.mem_manager = mem_manager

        # column configuration
        self.category_col = category_col
        self.phase_col = phase_col
        self.problem_col = problem_col
        self.solution_col = solution_col

        self.phase_threshold = phase_match_threshold
        self.category_threshold = category_match_threshold

    # -------------------------------------------------
    def _correct_spelling(self, text):
        try:
            return str(TextBlob(text).correct())
        except Exception:
            return text

    # -------------------------------------------------
    def _load_embeddings(self, mem_id):
        if supabase_store.is_enabled():
            payload = supabase_store.load_embeddings(mem_id)
            if payload:
                return payload

        path = (
            Path(self.mem_manager.base)
            / "memories"
            / f"{mem_id}_embeddings.json"
        )

        if not path.exists():
            return None

        payload = json.loads(path.read_text())

        if not isinstance(payload, dict):
            return None
        if "embeddings" not in payload or "row_ids" not in payload:
            return None

        return payload

    # -------------------------------------------------
    def _extract_context_hints(self, query, df):
        q = query.lower()
        matched_phase = None
        matched_category = None

        phases = []
        categories = []
        if self.phase_col in df.columns:
            phases = [
                str(x) for x in pd.unique(df[self.phase_col].dropna())
                if str(x).strip()
            ]

        if self.category_col in df.columns:
            categories = [
                str(x) for x in pd.unique(df[self.category_col].dropna())
                if str(x).strip()
            ]

        for p in phases:
            if fuzz.partial_ratio(p.lower(), q) >= self.phase_threshold:
                matched_phase = p
                break

        for c in categories:
            if fuzz.partial_ratio(c.lower(), q) >= self.category_threshold:
                matched_category = c
                break

        return matched_phase, matched_category

    # -------------------------------------------------
    def query_memory(
        self,
        mem_id,
        query,
        min_score=0.25,
        spell_correction=True,
        hard_limit=None,
        enforce_context=False,
        weight_text=0.70,
        weight_phase=0.15,
        weight_category=0.15,
        fallback_k=3,
        search_columns=None,
    ):
        if not self.emb_engine:
            return pd.DataFrame()

        q_text = self._correct_spelling(query) if spell_correction else query

        df = self.mem_manager.load_memory_dataframe(mem_id)
        if df is None or df.empty:
            return pd.DataFrame()

        if search_columns:
            valid_cols = [col for col in search_columns if col in df.columns]
            if not valid_cols:
                return pd.DataFrame()
            row_ids = list(df.index)
            texts = [_row_text(row, valid_cols) for _, row in df.iterrows()]
            usable = [(idx, text) for idx, text in zip(row_ids, texts) if text]
            if not usable:
                return pd.DataFrame()
            row_ids = [idx for idx, _ in usable]
            embeddings = self.emb_engine.embed_texts([text for _, text in usable])
        else:
            payload = self._load_embeddings(mem_id)
            if payload is None:
                raise FileNotFoundError(
                    f"Embeddings for memory '{mem_id}' not found."
                )

            embeddings = payload["embeddings"]
            row_ids = payload["row_ids"]

        matched_phase, matched_category = self._extract_context_hints(q_text, df)

        candidate_idxs = list(range(len(embeddings)))

        if enforce_context:
            filtered = []
            for emb_idx in candidate_idxs:
                df_idx = row_ids[emb_idx]
                row = df.loc[df_idx]

                if matched_phase and str(row[self.phase_col]).lower() != matched_phase.lower():
                    continue
                if matched_category and str(row[self.category_col]).lower() != matched_category.lower():
                    continue

                filtered.append(emb_idx)

            candidate_idxs = filtered or candidate_idxs

        q_emb = self.emb_engine.embed_texts([q_text])[0]

        scored = []
        for emb_idx in candidate_idxs:
            df_idx = row_ids[emb_idx]
            row = df.loc[df_idx]

            text_score = _cosine_similarity(q_emb, embeddings[emb_idx])

            phase_bonus = 0.0
            if matched_phase:
                if self.phase_col not in row.index:
                    phase_bonus = 0.0
                else:
                    s = fuzz.partial_ratio(
                        str(row[self.phase_col]).lower(),
                        matched_phase.lower(),
                    )
                    phase_bonus = 1.0 if s >= 85 else 0.5 if s >= 65 else 0.0

            category_bonus = 0.0
            if matched_category:
                if self.category_col not in row.index:
                    category_bonus = 0.0
                else:
                    s = fuzz.partial_ratio(
                        str(row[self.category_col]).lower(),
                        matched_category.lower(),
                    )
                    category_bonus = 1.0 if s >= 85 else 0.5 if s >= 65 else 0.0

            final_score = (
                weight_text * text_score
                + weight_phase * phase_bonus
                + weight_category * category_bonus
            )

            scored.append({
                "df_idx": df_idx,
                "TextScore": round(text_score, 4),
                "PhaseBonus": phase_bonus,
                "CategoryBonus": category_bonus,
                "FinalScore": round(final_score, 4),
            })

        scored_df = pd.DataFrame(scored).sort_values(
            "FinalScore", ascending=False
        )

        matches = scored_df[scored_df["FinalScore"] >= min_score]

        if matches.empty:
            matches = scored_df.sort_values(
                "TextScore", ascending=False
            ).head(fallback_k)

        if hard_limit:
            matches = matches.head(hard_limit)

        out_rows = []
        for _, r in matches.iterrows():
            row = df.loc[r["df_idx"]]
            rec = row.to_dict()
            rec.update(r.to_dict())
            rec["Citation"] = f"R{len(out_rows) + 1}"
            out_rows.append(rec)

        return pd.DataFrame(out_rows)


    # -------------------------------------------------
    def generate_structured_insights(self, df: pd.DataFrame):
        """
        Generate structured insights from recall results.
        Deterministic, safe, no LLM required.
        """

        if df is None or df.empty:
            return {
                "matches": 0,
                "summary": "No relevant historical records found.",
                "top_machine_types": [],
                "top_applications": [],
                "common_problems": [],
                "common_solutions": []
            }

        def top_values(series, k=5):
            if series is None:
                return []
            return (
                series.dropna()
                .astype(str)
                .value_counts()
                .head(k)
                .index.tolist()
            )

        insights = {
            "matches": len(df),
            "summary": f"{len(df)} relevant historical records found.",
            "top_machine_types": top_values(df.get(self.category_col)),
            "top_applications": top_values(df.get(self.phase_col)),
            "common_problems": (
                (df.get(self.problem_col) if df.get(self.problem_col) is not None else pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .head(5)
                .tolist()
            ),
            "common_solutions": (
                (df.get(self.solution_col) if df.get(self.solution_col) is not None else pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .head(5)
                .tolist()
            )
        }

        return insights
    def generate_natural_language_answer(
        self,
        insights: dict,
        query: str,
        template: dict | None = None
    ) -> str:
    
        # -------------------------------
        # Fallback: no data
        # -------------------------------
        if not insights or insights.get("matches", 0) == 0:
            return (
                f"I could not find relevant past projects related to your query: "
                f"'{query}'."
            )
    
        # -------------------------------
        # If NO template → keep OLD behavior
        # -------------------------------
        if not template:
            machines = ", ".join(insights.get("top_machine_types", []))
            applications = ", ".join(insights.get("top_applications", []))
    
            problems = insights.get("common_problems", [])
            solutions = insights.get("common_solutions", [])
    
            text = []
            text.append(
                f"Based on {insights['matches']} similar historical projects related to "
                f"**{applications}**, mainly involving **{machines}**, the following "
                f"recurring problems were identified:"
            )
    
            for p in problems:
                text.append(f"- {p}")
    
            if solutions:
                text.append(
                    "\nTo address these issues, the following solutions and lessons learned were applied:"
                )
                for s in solutions:
                    text.append(f"- {s}")
    
            text.append(
                "\n**Summary:** Similar projects show that accurate layout definition, "
                "realistic installation effort estimation, and early alignment with "
                "clients and suppliers are critical to avoid cost overruns and delays."
            )
    
            return "\n".join(text)
    
        # -------------------------------
        # TEMPLATE-DRIVEN OUTPUT
        # -------------------------------
        sections = template.get("sections", [])
        tone = template.get("tone", "simple")
        length = template.get("length", "short")
    
        machines = ", ".join(insights.get("top_machine_types", []))
        applications = ", ".join(insights.get("top_applications", []))
        problems = insights.get("common_problems", [])
        solutions = insights.get("common_solutions", [])
    
        output = []
    
        for section in sections:
            title = section.strip().lower()
    
            if title == "problem":
                output.append("### Problem")
                for p in problems[:5]:
                    output.append(f"- {p}")
    
            elif title == "solution":
                output.append("### Solution")
                for s in solutions[:5]:
                    output.append(f"- {s}")
    
            elif title == "key takeaways":
                output.append("### Key Takeaways")
                output.append(
                    f"- Based on **{insights['matches']}** similar projects in **{applications}**"
                )
                output.append("- Early planning and stakeholder alignment are critical")
                output.append("- Clear scope definition reduces execution risk")
    
            elif title == "context":
                output.append("### Context")
                output.append(f"- Applications: {applications}")
                output.append(f"- Machine types: {machines}")
    
            else:
                # Custom section name
                output.append(f"### {section}")
                output.append(
                    "Relevant insights derived from historical project records."
                )
    
        return "\n".join(output)

    # -------------------------------------------------
    def filter_memory(
        self,
        mem_id: str,
        column: str,
        value: str,
        exact: bool = False
    ) -> pd.DataFrame:
    
        df = self.mem_manager.load_memory_dataframe(mem_id)
        if df is None or df.empty:
            return pd.DataFrame()
    
        if column != "All columns" and column not in df.columns:
            return pd.DataFrame()

        if column == "All columns":
            joined = df.astype(str).agg(" | ".join, axis=1)
            mask = joined.str.contains(value, case=False, na=False)
        elif exact:
            mask = df[column].astype(str).str.lower() == value.lower()
        else:
            mask = df[column].astype(str).str.contains(
                value, case=False, na=False
            )

        out = df[mask].reset_index(drop=True)
        if not out.empty:
            out["Citation"] = [f"F{i + 1}" for i in range(len(out))]
        return out




    def generate_llm_summary(
        self,
        insights: dict,
        query: str,
        template: dict,
        instructions: str,
        result_rows: pd.DataFrame | None = None,
    ) -> str:
    
        client = OpenAI()
    
        grounded_rows = ""
        if result_rows is not None and not result_rows.empty:
            safe_cols = [
                col for col in result_rows.columns
                if col not in {"__semantic_text__", "TextScore", "PhaseBonus", "CategoryBonus"}
            ]
            snippets = []
            for i, (_, row) in enumerate(result_rows.head(8).iterrows(), start=1):
                citation = row.get("Citation", f"R{i}")
                values = []
                for col in safe_cols[:12]:
                    val = str(row.get(col, "")).strip()
                    if val:
                        values.append(f"{col}: {val[:500]}")
                snippets.append(f"[{citation}] " + " | ".join(values))
            grounded_rows = "\n".join(snippets)

        prompt = f"""
    You are an expert analyst.
    
    User query:
    {query}
    
    Historical insights:
    - Total matches: {insights['matches']}
    - Applications: {', '.join(insights.get('top_applications', []))}
    - Machine types: {', '.join(insights.get('top_machine_types', []))}
    
    Common problems:
    {chr(10).join(insights.get('common_problems', []))}
    
    Common solutions:
    {chr(10).join(insights.get('common_solutions', []))}

    Grounded source records:
    {grounded_rows}
    
    User instructions:
    {instructions}
    
    Required structure:
    Sections: {', '.join(template.get('sections', []))}
    Tone: {template.get('tone')}
    Length: {template.get('length')}
    
    Write a clear, outcome-driven summary based only on the source records.
    Include citations like [R1] or [R2] for claims tied to retrieved records.
    """
    
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
    
        return resp.choices[0].message.content.strip()
