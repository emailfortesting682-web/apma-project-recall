# modules/data_handler.py

import pandas as pd
import re

# Final required schema
REQUIRED_COLS = [
    'COMMESSA',
    'CLIENTE',
    'ANNO',
    'TIPO MACCHINA',
    'APPLICAZIONE',
    'TIPO PROBLEMA',
    'DESCRIZIONE',
    'SOLUZIONE LESSON LEARNED',
    'DATA INSERIMENTO',
    'RCPRD',
    'REPORT CANTIERE',
    'CONCERNED DEPARTMENTS',
    'REPORT RIUNIONE CHIUSURA PROGETTO'
]

# Legacy → New column mapping
LEGACY_MAP = {
    'project category': 'TIPO MACCHINA',
    'project reference': 'CLIENTE',
    'phase': 'APPLICAZIONE',
    'problems encountered': 'DESCRIZIONE',
    'solutions adopted': 'SOLUZIONE LESSON LEARNED'
}


def normalize(col: str) -> str:
    """Normalize column name for fuzzy matching"""
    return re.sub(r'[^a-z0-9]', '', col.lower())


class DataHandler:
    @staticmethod
    def read_and_validate(uploaded_file, required_cols=None):
        """
        Read CSV / Excel file, normalize headers, map legacy names,
        and validate against required schema.
        """

        if required_cols is None:
            required_cols = REQUIRED_COLS

        try:
            name = uploaded_file.name.lower()
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            return None, f"❌ Could not read file: {e}"

        if df.empty:
            return None, "❌ Uploaded file is empty."

        original_cols = list(df.columns)

        # Build normalized lookup
        normalized_lookup = {normalize(c): c for c in original_cols}

        rename_map = {}

        # 1️⃣ Handle legacy column names
        for legacy, new in LEGACY_MAP.items():
            norm_legacy = normalize(legacy)
            if norm_legacy in normalized_lookup:
                rename_map[normalized_lookup[norm_legacy]] = new

        # 2️⃣ Match required columns (fuzzy + exact)
        for rc in required_cols:
            if rc in original_cols:
                continue

            norm_rc = normalize(rc)

            if norm_rc in normalized_lookup:
                rename_map[normalized_lookup[norm_rc]] = rc
                continue

            # partial fuzzy match
            for norm_col, original in normalized_lookup.items():
                if norm_rc.startswith(norm_col) or norm_col.startswith(norm_rc):
                    rename_map[original] = rc
                    break

        if rename_map:
            df = df.rename(columns=rename_map)

        # 3️⃣ Final validation
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"❌ Missing required columns: {missing}"

        # 4️⃣ Enforce schema + sanitize
        df = df[required_cols].fillna("").astype(str)

        return df, None
