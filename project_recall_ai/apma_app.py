import os
import json
from io import BytesIO
from datetime import date

import pandas as pd
import streamlit as st
from streamlit import rerun

from modules import user_manager
from modules.data_handler import DataHandler
from modules.download_utils import export_csv, export_excel, export_pdf, export_word
from modules.embeddings_engine import EmbeddingsEngine
from modules.file_manager import MemoryManager
from modules.manual_config import load_config, save_config
from modules.recall_engine import RecallEngine
from modules.summary_parser import parse_summary_instructions
from modules.summary_templates import load_templates, save_templates
from modules.utils import ensure_data_dirs


try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

REQUIRED_COLS = [
    "COMMESSA",
    "CLIENTE",
    "ANNO",
    "TIPO MACCHINA",
    "APPLICAZIONE",
    "TIPO PROBLEMA",
    "DESCRIZIONE",
    "SOLUZIONE LESSON LEARNED",
    "DATA INSERIMENTO",
    "RCPRD",
    "REPORT CANTIERE",
    "CONCERNED DEPARTMENTS",
    "REPORT RIUNIONE CHIUSURA PROGETTO",
]

SYSTEM_COLS = {
    "__semantic_text__",
    "AddedBy",
    "AddedByName",
    "CreatedAt",
    "ModifiedBy",
    "ModifiedByName",
    "ModifiedAt",
    "df_idx",
    "TextScore",
    "PhaseBonus",
    "CategoryBonus",
    "FinalScore",
}

TRACEABILITY_COLS = ["AddedBy", "AddedByName", "CreatedAt", "ModifiedBy", "ModifiedByName", "ModifiedAt"]


def normalize(col: str) -> str:
    return col.lower().replace(" ", "").replace("_", "")


def build_semantic_text(df: pd.DataFrame) -> pd.Series:
    cfg = load_config()
    semantic_cols = [
        col
        for col, meta in cfg.items()
        if meta.get("type") in ("text", "select", "date") and col in df.columns
    ]
    if not semantic_cols:
        semantic_cols = list(df.columns)
    return df[semantic_cols].astype(str).fillna("").agg(" | ".join, axis=1)


def append_to_memory(mem_manager, memory_name, new_df):
    if memory_name in mem_manager.list_memories():
        existing = mem_manager.load_memory_dataframe(memory_name)
        return pd.concat([existing, new_df], ignore_index=True)
    return new_df


def current_username() -> str:
    user = st.session_state.get("user") or {}
    return " ".join(
        part for part in [user.get("first_name", ""), user.get("last_name", "")] if part
    ).strip() or user.get("id", "")


def add_traceability(df: pd.DataFrame, is_new_record: bool = True) -> pd.DataFrame:
    user = st.session_state.get("user") or {}
    now = pd.Timestamp.utcnow().isoformat()
    out = df.copy()
    if is_new_record:
        out["AddedBy"] = user.get("id", "")
        out["AddedByName"] = current_username()
        out["CreatedAt"] = now
    for col in ["AddedBy", "AddedByName", "CreatedAt"]:
        if col not in out.columns:
            out[col] = ""
    out["ModifiedBy"] = user.get("id", "")
    out["ModifiedByName"] = current_username()
    out["ModifiedAt"] = now
    return out


def infer_schema(df: pd.DataFrame) -> dict:
    schema = {}
    for col in df.columns:
        if col in SYSTEM_COLS:
            continue
        schema[col] = {"type": str(df[col].dtype), "required": col in REQUIRED_COLS}
    return schema


def get_existing_columns(mem_manager):
    cols = set()
    for mem in mem_manager.list_memories():
        df = mem_manager.load_memory_dataframe(mem)
        cols.update([c for c in df.columns if c not in SYSTEM_COLS])
    return sorted(cols)


def inject_professional_theme():
    st.markdown(
        """
        <style>
        :root {
            --apma-ink: #111827;
            --apma-blue: #0f4bd8;
            --apma-blue-hover: #0b3bb5;
            --apma-blue-soft: #eaf3ff;
            --apma-border: #e5e7eb;
            --apma-muted: #4b5563;
            --apma-bg: #f4f4f4;
            --apma-panel: #ffffff;
            --apma-nav: #faf9f8;
        }
        html, body, [data-testid="stAppViewContainer"] {
            background: var(--apma-bg);
            color: var(--apma-ink);
            font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .block-container {
            padding-top: 1.35rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }
        h1, h2, h3 {
            color: var(--apma-ink);
            letter-spacing: 0;
            font-family: "Segoe UI", system-ui, sans-serif;
        }
        [data-testid="stSidebar"] {
            background: var(--apma-nav);
            border-right: 1px solid var(--apma-border);
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: var(--apma-ink);
        }
        .apma-hero {
            border: 0;
            background: #ffffff;
            border-radius: 18px;
            padding: 0;
            margin-bottom: 22px;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
            display: flex;
            align-items: stretch;
            justify-content: space-between;
            overflow: hidden;
            min-height: 250px;
        }
        .apma-hero-copy {
            padding: 42px 48px;
            flex: 1 1 auto;
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-width: 0;
        }
        .apma-title {
            margin: 0;
            color: var(--apma-ink);
            font-size: 42px;
            line-height: 1.12;
            font-weight: 700;
            max-width: 680px;
        }
        .apma-subtitle {
            margin: 14px 0 0 0;
            color: var(--apma-muted);
            font-size: 18px;
            line-height: 1.55;
            max-width: 720px;
        }
        .apma-hero-art {
            position: relative;
            flex: 0 0 34%;
            min-width: 300px;
            background:
                radial-gradient(circle at 28% 30%, rgba(255, 122, 89, 0.42), transparent 22%),
                radial-gradient(circle at 70% 28%, rgba(80, 230, 255, 0.42), transparent 24%),
                radial-gradient(circle at 46% 72%, rgba(128, 214, 122, 0.42), transparent 25%),
                linear-gradient(135deg, #fff4e7 0%, #efe0ca 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        .apma-icon-cloud {
            position: relative;
            width: 310px;
            height: 205px;
        }
        .apma-app-icon {
            position: absolute;
            width: 52px;
            height: 52px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-weight: 750;
            font-size: 20px;
            box-shadow: 0 14px 24px rgba(17, 24, 39, 0.18);
            transform: rotate(-6deg);
        }
        .apma-app-icon::after {
            content: "";
            position: absolute;
            inset: 8px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.26);
        }
        .apma-i1 { left: 10px; top: 62px; background: linear-gradient(135deg, #f25022, #ff8a00); }
        .apma-i2 { left: 90px; top: 18px; background: linear-gradient(135deg, #7f39fb, #ff5ca8); }
        .apma-i3 { left: 166px; top: 48px; background: linear-gradient(135deg, #0078d4, #50e6ff); transform: rotate(7deg); }
        .apma-i4 { left: 234px; top: 22px; background: linear-gradient(135deg, #6264a7, #8b5cf6); }
        .apma-i5 { left: 72px; top: 126px; background: linear-gradient(135deg, #0f6cbd, #3b82f6); transform: rotate(5deg); }
        .apma-i6 { left: 154px; top: 135px; background: linear-gradient(135deg, #107c10, #80d67a); }
        .apma-i7 { left: 238px; top: 116px; background: linear-gradient(135deg, #8764b8, #c084fc); transform: rotate(8deg); }
        .apma-float {
            position: absolute;
            width: 70px;
            height: 18px;
            background: rgba(17, 24, 39, 0.08);
            border-radius: 999px;
            filter: blur(6px);
        }
        .apma-s1 { left: 5px; top: 123px; }
        .apma-s2 { left: 92px; top: 84px; }
        .apma-s3 { left: 170px; top: 108px; }
        .apma-s4 { left: 236px; top: 80px; }
        .apma-help {
            border: 0;
            background: var(--apma-blue-soft);
            padding: 10px 12px;
            color: #1c2b33;
            margin: 6px 0 12px 0;
            border-radius: 8px;
            line-height: 1.35;
            font-size: 14px;
        }
        .apma-card {
            border: 1px solid var(--apma-border);
            border-top: 4px solid var(--apma-blue);
            border-radius: 10px;
            background: var(--apma-panel);
            padding: 18px 18px;
            min-height: 108px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06);
        }
        .apma-card-title {
            color: var(--apma-muted);
            font-size: 12px;
            font-weight: 600;
            text-transform: none;
            margin-bottom: 6px;
        }
        .apma-card-value {
            color: var(--apma-ink);
            font-size: 24px;
            font-weight: 600;
            letter-spacing: 0;
        }
        .apma-workflow {
            border: 1px solid var(--apma-border);
            border-radius: 10px;
            padding: 16px 18px;
            background: var(--apma-panel);
            margin-bottom: 10px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
            color: var(--apma-ink);
            font-size: 14px;
        }
        .apma-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 0 0 12px 0;
        }
        .apma-mark {
            display: grid;
            grid-template-columns: repeat(2, 13px);
            grid-template-rows: repeat(2, 13px);
            gap: 3px;
            flex: 0 0 auto;
        }
        .apma-mark span {
            display: block;
            border-radius: 2px;
        }
        .apma-mark span:nth-child(1) { background: #0078d4; }
        .apma-mark span:nth-child(2) { background: #50e6ff; }
        .apma-mark span:nth-child(3) { background: #107c10; }
        .apma-mark span:nth-child(4) { background: #8764b8; }
        .apma-brand-name {
            font-weight: 600;
            color: var(--apma-ink);
            line-height: 1.1;
        }
        .apma-brand-sub {
            color: var(--apma-muted);
            font-size: 12px;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 7px;
            font-weight: 600;
            border: 1px solid transparent;
            background: #143bd6;
            color: #ffffff;
            padding: 0.52rem 1.05rem;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: var(--apma-blue-hover);
            color: #ffffff;
            border-color: transparent;
        }
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input {
            background: #ffffff !important;
            color: var(--apma-ink) !important;
            border: 1px solid #8a8886 !important;
        }
        .stTextInput input:focus,
        .stTextArea textarea:focus,
        .stNumberInput input:focus {
            border-color: var(--apma-blue) !important;
            box-shadow: 0 0 0 1px var(--apma-blue) !important;
        }
        [data-baseweb="select"] > div {
            background: #ffffff !important;
            color: var(--apma-ink) !important;
            border-color: #8a8886 !important;
        }
        [data-baseweb="radio"] {
            color: var(--apma-ink) !important;
        }
        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        textarea {
            border-radius: 2px !important;
        }
        .stDataFrame {
            border-radius: 4px;
            overflow: hidden;
            border: 1px solid var(--apma-border);
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--apma-border);
            border-radius: 10px;
            background: #ffffff;
        }
        [data-testid="stAlert"] {
            border-radius: 10px;
        }
        .apma-search-status {
            color: var(--apma-muted);
            font-size: 15px;
            line-height: 1.4;
            padding: 6px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .apma-search-icon {
            width: 18px;
            height: 18px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--apma-blue);
            flex: 0 0 auto;
        }
        .apma-search-icon.scan {
            animation: apma-scan 1.25s ease-in-out infinite;
        }
        .apma-spinner {
            width: 16px;
            height: 16px;
            border: 2px solid #d1d5db;
            border-top-color: var(--apma-blue);
            border-radius: 999px;
            animation: apma-spin 0.85s linear infinite;
            flex: 0 0 auto;
        }
        .apma-done-dot {
            width: 9px;
            height: 9px;
            background: #107c10;
            border-radius: 999px;
            box-shadow: 0 0 0 4px rgba(16, 124, 16, 0.12);
            flex: 0 0 auto;
        }
        @keyframes apma-spin {
            to { transform: rotate(360deg); }
        }
        @keyframes apma-scan {
            0%, 100% { transform: translateX(0) rotate(-8deg); opacity: 0.75; }
            50% { transform: translateX(5px) rotate(8deg); opacity: 1; }
        }
        @media (max-width: 900px) {
            .apma-hero {
                flex-direction: column;
            }
            .apma-hero-copy {
                padding: 30px 28px;
            }
            .apma-hero-art {
                min-width: 100%;
                min-height: 210px;
            }
            .apma-title {
                font-size: 34px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, description: str):
    st.markdown(
        f"""
        <div class="apma-hero">
            <div class="apma-hero-copy">
                <div class="apma-mark" aria-hidden="true"><span></span><span></span><span></span><span></span></div>
                <div class="apma-title">{title}</div>
                <p class="apma-subtitle">{description}</p>
            </div>
            <div class="apma-hero-art" aria-hidden="true">
                <div class="apma-icon-cloud">
                    <div class="apma-float apma-s1"></div>
                    <div class="apma-float apma-s2"></div>
                    <div class="apma-float apma-s3"></div>
                    <div class="apma-float apma-s4"></div>
                    <div class="apma-app-icon apma-i1">D</div>
                    <div class="apma-app-icon apma-i2">A</div>
                    <div class="apma-app-icon apma-i3">S</div>
                    <div class="apma-app-icon apma-i4">N</div>
                    <div class="apma-app-icon apma-i5">K</div>
                    <div class="apma-app-icon apma-i6">X</div>
                    <div class="apma-app-icon apma-i7">R</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def guidance(text: str):
    with st.expander("Need help?", expanded=False):
        st.write(text)


def set_search_status(container, text: str, state: str = "search"):
    if state == "thinking":
        icon = '<span class="apma-spinner" aria-hidden="true"></span>'
    elif state == "done":
        icon = '<span class="apma-done-dot" aria-hidden="true"></span>'
    else:
        icon = '<span class="apma-search-icon scan" aria-hidden="true">⌕</span>'
    container.markdown(
        f'<div class="apma-search-status">{icon}<span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, note: str):
    st.markdown(
        f"""
        <div class="apma-card">
            <div class="apma-card-title">{title}</div>
            <div class="apma-card-value">{value}</div>
            <div class="apma-subtitle">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_login():
    if not st.session_state.get("user"):
        st.warning("Please log in from the sidebar to use this page.")
        st.stop()


def memory_record_count(mem_manager, memory_id: str) -> int:
    try:
        return len(mem_manager.load_memory_dataframe(memory_id))
    except Exception:
        return 0


def save_memory_with_embeddings(mem_manager, emb_engine, name: str, df: pd.DataFrame, allowed_user_ids=None, audit_action="save_memory"):
    meta = mem_manager.create_or_update_memory(
        name,
        df,
        allowed_user_ids=allowed_user_ids,
        schema=infer_schema(df),
        audit_action=audit_action,
    )
    if emb_engine:
        emb_engine.index_dataframe(meta["memory_path"], df, id_prefix=meta["memory_id"])
    return meta


def sample_template_dataframe() -> pd.DataFrame:
    return pd.DataFrame([{col: "" for col in REQUIRED_COLS}])


def excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="APMA Template", index=False)
    return buffer.getvalue()


def export_json(df: pd.DataFrame, summary: str) -> bytes:
    payload = {
        "summary": summary,
        "records": df.fillna("").to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def run_hybrid_query(engine, memory_name, query, search_columns, top_k, semantic_weight, lexical_weight, structured_weight):
    if hasattr(engine, "hybrid_query_memory"):
        return engine.hybrid_query_memory(
            memory_name,
            query,
            search_columns=search_columns,
            top_k=top_k,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
            structured_weight=structured_weight,
        )
    return engine.query_memory(
        memory_name,
        query,
        search_columns=search_columns,
        hard_limit=top_k,
    )


def read_uploaded_flexible(uploaded_file):
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, header=None).fillna("").astype(str)
    return pd.read_excel(uploaded_file, header=None).fillna("").astype(str)


def clean_column_name(value, index: int) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return f"Column {index + 1}"
    return text


def make_unique_columns(columns: list[str]) -> list[str]:
    seen = {}
    unique = []
    for col in columns:
        base = str(col).strip() or "Column"
        count = seen.get(base, 0)
        unique.append(base if count == 0 else f"{base}_{count + 1}")
        seen[base] = count + 1
    return unique


def dataframe_from_header_choice(raw_df: pd.DataFrame, use_first_row: bool, columns: list[str]) -> pd.DataFrame:
    data = raw_df.iloc[1:].reset_index(drop=True) if use_first_row else raw_df.copy()
    data.columns = make_unique_columns(columns)
    return data.fillna("").astype(str)


def schema_columns_from_metadata(mem_manager, memory_name: str) -> list[str]:
    meta = mem_manager.get_memory_metadata(memory_name) if memory_name else {}
    schema = meta.get("schema") or {}
    return [col for col in schema.keys() if col not in SYSTEM_COLS]


def align_to_schema(df: pd.DataFrame, schema_cols: list[str], mapping: dict[str, str]) -> pd.DataFrame:
    aligned = pd.DataFrame()
    for target_col in schema_cols:
        source_col = mapping.get(target_col)
        if source_col and source_col in df.columns:
            aligned[target_col] = df[source_col]
        else:
            aligned[target_col] = ""
    return aligned


def clear_search_state():
    for key in ["last_result_df", "last_summary", "last_query", "last_result_memory"]:
        st.session_state.pop(key, None)


def go_to(page: str):
    st.session_state["workspace_nav"] = page
    rerun()


def permission_input(key_prefix: str):
    share_mode = st.radio(
        "Memory access",
        ["Shared workspace", "Only me", "Specific users"],
        horizontal=True,
        key=f"{key_prefix}_share_mode",
        help="Choose who can find and add records to this memory.",
    )
    if share_mode == "Shared workspace":
        return ["*"]
    if share_mode == "Only me":
        return [st.session_state["user"]["id"]]
    raw_ids = st.text_input(
        "Allowed user IDs",
        key=f"{key_prefix}_allowed_ids",
        help="Enter comma-separated user IDs that can access this memory.",
    )
    ids = [item.strip() for item in raw_ids.split(",") if item.strip()]
    owner_id = st.session_state["user"]["id"]
    return sorted(set(ids + [owner_id]))


def render_download_panel(df: pd.DataFrame, summary: str):
    st.markdown("### Export report")
    guidance("Choose whether the client needs the raw result table, the AI summary, or a complete report.")

    has_summary = bool(str(summary).strip())
    report_options = ["Only Table Results", "Full Report (Table + Summary)"]
    if has_summary:
        report_options.insert(1, "Only Summary Report")

    if not has_summary:
        st.info("No AI summary is available for this result, so summary-only export is hidden.")

    download_type = st.radio(
        "Report contents",
        report_options,
        horizontal=True,
        help="Controls which parts of the current search result are included in the exported file.",
    )
    format_choice = st.selectbox(
        "File format",
        ["CSV", "Excel", "PDF", "Word", "JSON"],
        key="download_format_choice",
        help="Select the format that is easiest for your client to review or archive.",
    )

    if download_type == "Only Table Results":
        export_df = df
        export_summary = ""
    elif download_type == "Only Summary Report":
        export_df = pd.DataFrame()
        export_summary = summary
    else:
        export_df = df
        export_summary = summary

    if format_choice == "CSV":
        data = export_csv(export_df, export_summary)
        st.download_button("Download CSV", data, "report.csv", "text/csv")
    elif format_choice == "Excel":
        data = export_excel(export_df, export_summary)
        st.download_button(
            "Download Excel",
            data,
            "report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    elif format_choice == "PDF":
        data = export_pdf(export_df, export_summary)
        st.download_button("Download PDF", data, "report.pdf", "application/pdf")
    elif format_choice == "Word":
        data = export_word(export_df, export_summary)
        st.download_button(
            "Download Word",
            data,
            "report.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    elif format_choice == "JSON":
        data = export_json(export_df, export_summary)
        st.download_button("Download JSON", data, "report.json", "application/json")


st.set_page_config(
    page_title="APMA - AI Project Memory Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_data_dirs()
inject_professional_theme()

if "user" not in st.session_state:
    st.session_state["user"] = None

mem_manager = MemoryManager(data_dir="data", current_user=st.session_state.get("user"))

emb_engine = None
if OPENAI_API_KEY:
    try:
        emb_engine = EmbeddingsEngine()
    except Exception:
        emb_engine = None

recall_engine = RecallEngine(
    emb_engine=emb_engine,
    mem_manager=mem_manager,
    category_col="TIPO MACCHINA",
    phase_col="APPLICAZIONE",
    problem_col="DESCRIZIONE",
    solution_col="SOLUZIONE LESSON LEARNED",
)


# Sidebar authentication and navigation
st.sidebar.markdown(
    """
    <div class="apma-brand">
        <div class="apma-mark" aria-hidden="true"><span></span><span></span><span></span><span></span></div>
        <div>
            <div class="apma-brand-name">APMA</div>
            <div class="apma-brand-sub">Project Memory</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()

if st.session_state["user"] is None:
    st.sidebar.subheader("Account access")
    auth_mode = st.sidebar.radio(
        "Access mode",
        ["Login", "Create account"],
        help="Create an account for first-time users, or log in with an existing ID number and password.",
    )

    if auth_mode == "Create account":
        fn = st.sidebar.text_input("First name", help="Used only to identify the logged-in user in the app.")
        ln = st.sidebar.text_input("Last name", help="Used only to identify the logged-in user in the app.")
        uid = st.sidebar.text_input("ID Number", help="Unique ID used as the account login.")
        pw = st.sidebar.text_input("Password", type="password", help="Use at least 6 characters.")
        if st.sidebar.button("Create account", help="Create a new APMA user profile."):
            ok, msg = user_manager.create_user(fn, ln, uid, pw)
            st.sidebar.success(msg) if ok else st.sidebar.error(msg)
    else:
        uid = st.sidebar.text_input("ID Number", help="Enter the ID number used when the account was created.")
        pw = st.sidebar.text_input("Password", type="password", help="Enter the account password.")
        if st.sidebar.button("Login", help="Sign in and unlock APMA features."):
            ok, msg, profile = user_manager.authenticate(uid, pw)
            if ok:
                st.session_state["user"] = profile
                rerun()
            st.sidebar.error(msg)
else:
    u = st.session_state["user"]
    st.sidebar.success(f"Signed in as {u['first_name']} {u['last_name']}")
    if st.sidebar.button("Logout", help="End the current session."):
        st.session_state["user"] = None
        rerun()

st.sidebar.divider()
mode = st.sidebar.radio(
    "Workspace",
    ["Dashboard", "Data Upload", "Manual Entry", "Search & Insights", "Settings"],
    key="workspace_nav",
    help="Follow the workflow from status review to data capture, search, reporting, and configuration.",
)


if mode == "Dashboard":
    if not st.session_state.get("user"):
        page_header(
            "Welcome to APMA",
            "Project memory, search, and reports in one workspace.",
        )
        st.info("Log in from the sidebar to continue.")

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Capture", "Upload", "Import CSV or Excel project records")
        with c2:
            metric_card("Recall", "Search", "Find similar historical cases with AI")
        with c3:
            metric_card("Report", "Export", "Generate summaries and client-ready reports")

        st.markdown("### Workflow")
        public_steps = [
            ("Centralize", "Store problems, solutions, and lessons learned."),
            ("Search", "Find similar historical records."),
            ("Report", "Export summaries and tables."),
        ]
        for title, body in public_steps:
            st.markdown(f'<div class="apma-workflow"><strong>{title}</strong><br>{body}</div>', unsafe_allow_html=True)
        st.stop()

    page_header(
        "Project Memory Dashboard",
        "Status, shortcuts, and saved memories.",
    )

    memories = mem_manager.list_memories()
    total_records = sum(memory_record_count(mem_manager, mem) for mem in memories)

    c1, c2 = st.columns(2)
    with c1:
        metric_card("Saved memories", str(len(memories)), "Knowledge bases available for search")
    with c2:
        metric_card("Stored records", str(total_records), "Rows available across all memories")

    st.markdown("### Workflow")
    steps = [
        ("1. Add data", "Upload a file or enter records manually."),
        ("2. Save memory", "Create or append a searchable memory."),
        ("3. Search", "Find matches and generate a report."),
        ("4. Configure", "Adjust fields and summary templates."),
    ]
    for title, body in steps:
        st.markdown(f'<div class="apma-workflow"><strong>{title}</strong><br>{body}</div>', unsafe_allow_html=True)

    st.markdown("### Existing memories")
    if memories:
        rows = [{"Memory": mem, "Records": memory_record_count(mem_manager, mem)} for mem in memories]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        export_mem = st.selectbox(
            "Export memory",
            memories,
            help="Download the complete raw repository for backup or review.",
        )
        raw_df = mem_manager.load_memory_dataframe(export_mem)
        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button(
                "Download raw CSV",
                raw_df.to_csv(index=False).encode("utf-8"),
                f"{export_mem}.csv",
                "text/csv",
            )
        with e2:
            st.download_button(
                "Download raw Excel",
                excel_bytes(raw_df),
                f"{export_mem}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with e3:
            st.download_button(
                "Download raw JSON",
                export_json(raw_df, ""),
                f"{export_mem}.json",
                "application/json",
            )
    else:
        if st.session_state.get("user"):
            guidance("No memories found yet. Use Upload project file or Add manual record to create the first project memory.")
        else:
            guidance("Log in from the sidebar to create and search project memories.")


elif mode == "Data Upload":
    page_header(
        "Data Upload",
        "Import files and save searchable project memory.",
    )
    require_login()
    guidance("Create a new memory from any structured file, or append to an existing memory using its saved column structure.")

    memories = mem_manager.list_memories()
    file_mem_mode = st.radio(
        "Upload destination",
        ["Create new memory", "Append to existing memory"],
        horizontal=True,
        key="file_mem_mode",
        help="New memories can define their own columns. Existing memories use their saved structure.",
    )

    uploaded = st.file_uploader(
        "Upload CSV or Excel file",
        ["csv", "xlsx"],
        help="Accepted formats are .csv and .xlsx.",
    )

    if uploaded:
        try:
            raw_df = read_uploaded_flexible(uploaded)
        except Exception as exc:
            st.error(f"Could not read file: {exc}")
            st.stop()

        if raw_df.empty:
            st.warning("The uploaded file is empty.")
            st.stop()

        if file_mem_mode == "Create new memory":
            mem_name = st.text_input(
                "New memory name",
                key="file_new_memory_name",
                help="Use a clear name for this knowledge repository.",
            )
            allowed_user_ids = permission_input("file_memory")

            first_row = raw_df.iloc[0].tolist()
            detected_cols = make_unique_columns([
                clean_column_name(value, idx) for idx, value in enumerate(first_row)
            ])

            st.markdown("### Detected column headers")
            st.dataframe(pd.DataFrame({"Detected column": detected_cols}), hide_index=True, width="stretch")

            use_first_row = st.radio(
                "Should the first row be treated as column headers?",
                ["Yes", "No"],
                horizontal=True,
                key="use_first_row_headers",
                help="Choose Yes if the first row contains column names. Choose No if it is a data row.",
            ) == "Yes"

            default_cols = detected_cols if use_first_row else [f"Column {i + 1}" for i in range(raw_df.shape[1])]
            edit_cols = st.radio(
                "Are these column names correct?",
                ["Yes, continue", "No, I want to edit them"],
                horizontal=True,
                key="edit_detected_columns",
            )

            final_cols = []
            if edit_cols == "No, I want to edit them":
                st.markdown("### Edit column names")
                col_widgets = st.columns(2)
                for idx, col_name in enumerate(default_cols):
                    with col_widgets[idx % 2]:
                        final_cols.append(
                            st.text_input(
                                f"Column {idx + 1}",
                                value=col_name,
                                key=f"detected_col_{idx}",
                            )
                        )
            else:
                final_cols = default_cols

            final_cols = make_unique_columns([clean_column_name(col, idx) for idx, col in enumerate(final_cols)])
            preview_df = dataframe_from_header_choice(raw_df, use_first_row, final_cols)

            st.warning(
                "Please ensure that future uploads for this memory repository use the same column header names and structure. "
                "Consistent column names improve data quality, retrieval accuracy, and knowledge organization."
            )
            st.markdown("### Import preview")
            st.dataframe(preview_df.head(20), width="stretch")

            if st.button("Create memory", help="Save this file and register these columns as the official memory structure."):
                if not mem_name:
                    st.error("Please enter a memory name.")
                    st.stop()
                preview_df = add_traceability(preview_df, is_new_record=True)
                preview_df["__semantic_text__"] = build_semantic_text(preview_df)
                save_memory_with_embeddings(
                    mem_manager,
                    emb_engine,
                    mem_name,
                    preview_df,
                    allowed_user_ids=allowed_user_ids,
                    audit_action="create_memory_from_detected_schema",
                )
                st.success(f"Memory '{mem_name}' created with {len(final_cols)} registered columns.")

        else:
            if not memories:
                st.warning("No existing memories are available yet.")
                st.stop()

            mem_name = st.selectbox(
                "Select existing memory",
                memories,
                key="file_existing_memory",
                help="The upload will be checked against this memory's saved column structure.",
            )
            schema_cols = schema_columns_from_metadata(mem_manager, mem_name)
            if not schema_cols:
                existing_df = mem_manager.load_memory_dataframe(mem_name)
                schema_cols = [col for col in existing_df.columns if col not in SYSTEM_COLS]

            first_row = raw_df.iloc[0].tolist()
            detected_cols = make_unique_columns([
                clean_column_name(value, idx) for idx, value in enumerate(first_row)
            ])

            st.markdown("### Detected upload columns")
            st.dataframe(pd.DataFrame({"Detected column": detected_cols}), hide_index=True, width="stretch")
            with st.expander("Saved memory structure", expanded=False):
                st.dataframe(pd.DataFrame({"Expected column": schema_cols}), hide_index=True, width="stretch")

            use_first_row = st.radio(
                "Should the first row be treated as column headers?",
                ["Yes", "No"],
                horizontal=True,
                key="append_use_first_row_headers",
            ) == "Yes"
            upload_cols = detected_cols if use_first_row else [f"Column {i + 1}" for i in range(raw_df.shape[1])]
            upload_df = dataframe_from_header_choice(raw_df, use_first_row, upload_cols)

            missing = [col for col in schema_cols if col not in upload_df.columns]
            extra = [col for col in upload_df.columns if col not in schema_cols]

            if missing or extra:
                st.warning("Column mismatch detected. Review or map the uploaded columns before appending.")
                if missing:
                    st.write(f"Missing expected columns: {', '.join(missing)}")
                if extra:
                    st.write(f"Extra uploaded columns: {', '.join(extra)}")

                mapping = {}
                st.markdown("### Map uploaded columns")
                available_sources = ["-- leave blank --"] + list(upload_df.columns)
                for target_col in schema_cols:
                    default_index = available_sources.index(target_col) if target_col in available_sources else 0
                    selected_source = st.selectbox(
                        target_col,
                        available_sources,
                        index=default_index,
                        key=f"map_{target_col}",
                    )
                    mapping[target_col] = "" if selected_source == "-- leave blank --" else selected_source
                append_df = align_to_schema(upload_df, schema_cols, mapping)
            else:
                append_df = upload_df[schema_cols].copy()

            st.markdown("### Append preview")
            st.dataframe(append_df.head(20), width="stretch")

            allowed_user_ids = mem_manager.get_memory_metadata(mem_name).get("allowed_user_ids", ["*"])
            if st.button("Append to memory", help="Append these rows using the memory's registered structure."):
                append_df = add_traceability(append_df, is_new_record=True)
                append_df["__semantic_text__"] = build_semantic_text(append_df)
                final_df = append_to_memory(mem_manager, mem_name, append_df)
                save_memory_with_embeddings(
                    mem_manager,
                    emb_engine,
                    mem_name,
                    final_df,
                    allowed_user_ids=allowed_user_ids,
                    audit_action="append_with_schema_validation",
                )
                st.success(f"Data appended to memory '{mem_name}'.")


elif mode == "Manual Entry":
    page_header(
        "Manual Entry",
        "Add individual lessons learned.",
    )
    require_login()
    guidance("Add one or more rows to the pending list, review them, then save the batch to a memory.")

    config = load_config()
    memories = mem_manager.list_memories()
    manual_data = {}

    mem_mode = st.radio(
        "Save manual entries to",
        ["Create new memory", "Append to existing memory"],
        horizontal=True,
        help="Choose whether this batch starts a new memory or extends an existing one.",
    )

    target_memory = None
    if mem_mode == "Create new memory":
        target_memory = st.text_input("New memory name", key="manual_new_memory_name")
    else:
        if memories:
            target_memory = st.selectbox("Select memory", memories, key="manual_existing_memory")
        else:
            st.warning("No existing memories are available yet.")

    if mem_mode == "Create new memory":
        allowed_user_ids = permission_input("manual_memory")
    else:
        allowed_user_ids = (
            mem_manager.get_memory_metadata(target_memory).get("allowed_user_ids", ["*"])
            if target_memory else ["*"]
        )

    def render_manual_field(field: str):
        meta = config[field]
        key = f"manual_{field}"
        help_text = f"Enter the value for {field}. This field is included in saved project memory."
        if meta["type"] == "text":
            if meta.get("multiline"):
                manual_data[field] = st.text_area(field, key=key, help=help_text, height=120)
            else:
                manual_data[field] = st.text_input(field, key=key, help=help_text)
        elif meta["type"] == "select":
            manual_data[field] = st.selectbox(field, meta.get("options", []), key=key, help=help_text)
        elif meta["type"] == "date":
            if meta.get("mode") == "year":
                y = st.selectbox(field, list(range(2000, date.today().year + 1)), key=key, help=help_text)
                manual_data[field] = str(y)
            else:
                d = st.date_input(field, key=key, help=help_text)
                manual_data[field] = d.isoformat()

    project_fields = ["COMMESSA", "CLIENTE", "ANNO", "TIPO MACCHINA", "APPLICAZIONE", "TIPO PROBLEMA"]
    detail_fields = ["DESCRIZIONE", "SOLUZIONE LESSON LEARNED"]
    report_fields = [
        "DATA INSERIMENTO",
        "RCPRD",
        "REPORT CANTIERE",
        "CONCERNED DEPARTMENTS",
        "REPORT RIUNIONE CHIUSURA PROGETTO",
    ]
    known_fields = set(project_fields + detail_fields + report_fields)
    extra_fields = [field for field in config if field not in known_fields]

    with st.form("manual_dynamic"):
        cols = st.columns(3)
        for idx, field in enumerate([f for f in project_fields if f in config]):
            with cols[idx % 3]:
                render_manual_field(field)

        for field in [f for f in detail_fields if f in config]:
            render_manual_field(field)

        cols = st.columns(2)
        for idx, field in enumerate([f for f in report_fields if f in config]):
            with cols[idx % 2]:
                render_manual_field(field)

        if extra_fields:
            st.markdown("#### Additional fields")
            cols = st.columns(2)
            for idx, field in enumerate(extra_fields):
                with cols[idx % 2]:
                    render_manual_field(field)

        submitted = st.form_submit_button("Add row", help="Add this manual record to the pending review table.")

    if submitted:
        st.session_state.setdefault("manual_rows", []).append(manual_data)
        st.success("Row added to pending entries.")

    if st.session_state.get("manual_rows"):
        st.markdown("### Pending manual entries")
        pending_df = pd.DataFrame(st.session_state["manual_rows"])
        st.dataframe(pending_df, width="stretch")

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Save manual entries", help="Save all pending rows and rebuild AI embeddings."):
                if not target_memory:
                    st.error("Please select or create a memory.")
                else:
                    df_manual = pending_df.copy()
                    for col in REQUIRED_COLS:
                        if col not in df_manual.columns:
                            df_manual[col] = ""

                    df_manual = add_traceability(df_manual, is_new_record=True)
                    df_manual["__semantic_text__"] = build_semantic_text(df_manual)
                    final_df = append_to_memory(mem_manager, target_memory, df_manual)
                    save_memory_with_embeddings(
                        mem_manager,
                        emb_engine,
                        target_memory,
                        final_df,
                        allowed_user_ids=allowed_user_ids,
                        audit_action="manual_records",
                    )
                    st.session_state["manual_rows"] = []
                    st.success(f"Saved manual entries to memory '{target_memory}'.")
                    st.rerun()
        with c2:
            if st.button("Clear pending entries", help="Remove unsaved rows from the pending table."):
                st.session_state["manual_rows"] = []
                st.rerun()


elif mode == "Search & Insights":
    page_header(
        "Search & Insights",
        "Find similar records and export reports.",
    )
    require_login()
    guidance("Use semantic search for natural-language questions. Use structured filters when you know the exact field to inspect.")

    user_id = st.session_state["user"]["id"]
    templates = load_templates(user_id=user_id)
    mems = mem_manager.list_memories()

    if not mems:
        st.warning("No memories found. Add records from Data Upload or Manual Entry first.")
        st.stop()

    top1, top2 = st.columns(2)
    with top1:
        mem = st.selectbox("Memory", ["All memories"] + mems, help="Select one memory or search all accessible memories.")
    with top2:
        summary_template_name = st.selectbox(
            "Summary format",
            list(templates.keys()),
            key="query_summary_template",
            help="Controls how the AI summary is structured.",
        )

    context = (mem, summary_template_name)
    if st.session_state.get("last_search_context") != context:
        clear_search_state()
        st.session_state["last_search_context"] = context

    search_tab, filter_tab = st.tabs(["Hybrid Search", "Structured Filter Search"])

    with search_tab:
        if mem == "All memories":
            column_source_frames = [mem_manager.load_memory_dataframe(item) for item in mems]
            memory_df_for_columns = pd.concat(column_source_frames, ignore_index=True) if column_source_frames else pd.DataFrame()
        else:
            memory_df_for_columns = mem_manager.load_memory_dataframe(mem)
        searchable_columns = [col for col in memory_df_for_columns.columns if col not in SYSTEM_COLS]
        search_scope = st.multiselect(
            "Search scope",
            searchable_columns,
            default=[],
            help="Leave empty to search the full record, or choose specific columns for column-aware retrieval.",
        )
        q = st.text_area(
            "Question or problem description",
            placeholder="Example: installation delays caused by unclear layout requirements",
            help="Describe the issue in natural language. The app finds similar historical records.",
        )

        top_k = st.slider(
            "Results",
            min_value=3,
            max_value=20,
            value=10,
            help="Choose how many top-ranked hybrid results should be used.",
        )

        with st.expander("Retrieval balance", expanded=False):
            semantic_weight = st.slider("Meaning match", 0.0, 1.0, 0.45, 0.05)
            lexical_weight = st.slider("Keyword match", 0.0, 1.0, 0.30, 0.05)
            structured_weight = st.slider("Column/exact match", 0.0, 1.0, 0.25, 0.05)

        if st.button("Search memory", help="Run hybrid retrieval and generate a grounded summary."):
            if not emb_engine:
                st.error("Embeddings engine is not available. Check OPENAI_API_KEY in Streamlit secrets.")
                st.stop()
            if not q.strip():
                st.warning("Please enter a query.")
                st.stop()

            status_box = st.empty()
            set_search_status(status_box, "Looking into data...", "search")
            try:
                if mem == "All memories":
                    frames = []
                    for item in mems:
                        item_res = run_hybrid_query(
                            recall_engine,
                            item,
                            q,
                            search_scope or None,
                            top_k,
                            semantic_weight,
                            lexical_weight,
                            structured_weight,
                        )
                        if not item_res.empty:
                            item_res["Memory"] = item
                            frames.append(item_res)
                    res = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                    if not res.empty:
                        res = res.sort_values("HybridScore", ascending=False).head(top_k).reset_index(drop=True)
                else:
                    res = run_hybrid_query(
                        recall_engine,
                        mem,
                        q,
                        search_scope or None,
                        top_k,
                        semantic_weight,
                        lexical_weight,
                        structured_weight,
                    )
            except FileNotFoundError:
                status_box.empty()
                st.error("Embeddings were not found for this memory. Re-save or re-upload the memory to rebuild them.")
                st.stop()

            if res.empty:
                status_box.empty()
                st.info("No matching results found.")
                st.stop()

            res["Citation"] = [f"R{i + 1}" for i in range(len(res))]

            set_search_status(status_box, "Thinking...", "thinking")
            insights = recall_engine.generate_structured_insights(res)
            template = templates[summary_template_name]
            answer = recall_engine.generate_llm_summary(
                insights=insights,
                query=q,
                template=template,
                instructions=template.get("instructions", ""),
                result_rows=res,
            )
            set_search_status(status_box, "Done.", "done")

            st.session_state["last_result_df"] = res
            st.session_state["last_summary"] = answer
            st.session_state["last_query"] = q
            st.session_state["last_result_memory"] = mem

    with filter_tab:
        filterable_columns = ["All columns"] + [
            col for col in (
                memory_df_for_columns.columns if mem == "All memories"
                else mem_manager.load_memory_dataframe(mem).columns
            )
            if col not in SYSTEM_COLS
        ]
        c1, c2, c3 = st.columns([1, 2, 1])
        col = c1.selectbox("Filter by", filterable_columns, help="Choose the column to search within.")
        val = c2.text_input("Value", help="Enter the value or partial text to match.")
        exact = c3.checkbox("Exact match", False, help="Require an exact value instead of partial matching.")

        if st.button("Apply filter", help="Return records that match the selected structured filter."):
            if mem == "All memories":
                frames = []
                for item in mems:
                    item_df = recall_engine.filter_memory(item, col, val, exact)
                    if not item_df.empty:
                        item_df["Memory"] = item
                        frames.append(item_df)
                df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            else:
                df = recall_engine.filter_memory(mem, col, val, exact)
            if not df.empty:
                df["Citation"] = [f"F{i + 1}" for i in range(len(df))]
            st.session_state["last_result_df"] = df
            st.session_state["last_summary"] = ""
            st.session_state["last_result_memory"] = mem
            st.info(f"{len(df)} records found.")

    if "last_result_df" in st.session_state:
        df = st.session_state["last_result_df"]
        summary = st.session_state.get("last_summary", "")

        st.markdown("### Results")
        st.dataframe(df, width="stretch")

        if summary:
            st.markdown("### Analysis summary")
            st.markdown(summary)

        render_download_panel(df, summary)


elif mode == "Settings":
    page_header(
        "Settings",
        "Configure fields, summaries, and system status.",
    )
    require_login()

    fields_tab, templates_tab, system_tab = st.tabs(["Manual fields", "Summary templates", "Workspace"])

    with fields_tab:
        guidance("Manual fields control the form shown on the Manual Entry page. Required system columns cannot be deleted.")
        cfg = load_config()
        memory_cols = get_existing_columns(mem_manager)
        config_cols = list(cfg.keys())
        existing_cols = sorted({c for c in memory_cols + config_cols if c not in SYSTEM_COLS})
        existing_norm = {normalize(c): c for c in existing_cols}
        field_add = "__ADD_NEW__"

        field = st.selectbox(
            "Select field",
            list(cfg.keys()) + [field_add],
            key="settings_field_selector",
            format_func=lambda x: "Add new field" if x == field_add else x,
            help="Choose a field to edit, or add a new manual-entry field.",
        )

        if field == field_add:
            st.markdown("### Add field")
            col_choice = st.selectbox(
                "Choose existing column",
                ["None"] + existing_cols,
                help="Reuse a column already found in saved memories, or choose None to create a new field.",
            )
            custom_name = st.text_input("New column name", help="Enter a clear field name if you are not reusing an existing column.")
            new_type = st.selectbox("Field type", ["text", "select", "date"], help="Controls how this field appears in Manual Entry.")
            final_name = col_choice if col_choice != "None" else custom_name.strip()

            if st.button("Save field", help="Add this field to the manual-entry configuration."):
                if not final_name:
                    st.error("Please select or enter a column name.")
                elif col_choice == "None" and (
                    normalize(final_name) in existing_norm or normalize(final_name) in map(normalize, cfg.keys())
                ):
                    st.warning("Column already exists. Choose it from the existing-column list.")
                else:
                    cfg[final_name] = {"type": new_type}
                    save_config(cfg)
                    st.session_state.pop("settings_field_selector", None)
                    st.success(f"Field '{final_name}' added.")
                    st.rerun()
        else:
            if field not in cfg:
                st.warning("Invalid field selected. Please reselect.")
                st.stop()

            st.markdown("### Edit field")
            meta = cfg[field]
            new_field_name = st.text_input("Field name", value=field, help="Rename this manual-entry field.")
            meta["type"] = st.selectbox(
                "Field type",
                ["text", "select", "date"],
                index=["text", "select", "date"].index(meta["type"]),
                help="Controls the input type used in Manual Entry.",
            )

            if meta["type"] == "select":
                opts = st.text_area(
                    "Dropdown options",
                    "\n".join(meta.get("options", [])),
                    help="Enter one dropdown option per line.",
                )
                meta["options"] = [o.strip() for o in opts.splitlines() if o.strip()]

            if meta["type"] == "date":
                meta["mode"] = st.radio("Date mode", ["full", "year"], help="Use full date for exact entries or year for annual references.")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Save field changes", help="Update this field configuration."):
                    new_norm = normalize(new_field_name)
                    existing_norms = {normalize(c): c for c in cfg.keys() if c != field}
                    if new_norm in existing_norms:
                        st.error("A column with this name already exists.")
                        st.stop()
                    cfg[new_field_name] = meta
                    if new_field_name != field:
                        del cfg[field]
                    save_config(cfg)
                    st.success("Field updated.")
                    st.rerun()

            with col_b:
                if st.button("Delete field", help="Remove this field from the manual-entry form."):
                    if field in REQUIRED_COLS:
                        st.error("This is a required system column and cannot be deleted.")
                        st.stop()
                    save_config({k: v for k, v in cfg.items() if k != field})
                    st.warning(f"Field '{field}' deleted.")
                    st.rerun()

    with templates_tab:
        guidance("Summary templates define the structure, tone, and length of AI-generated client reports.")
        user_id = st.session_state["user"]["id"]
        templates = load_templates(user_id=user_id)
        template_names = list(templates.keys())

        if template_names:
            selected_template = st.selectbox(
                "Summary template",
                template_names,
                key="selected_summary_template",
                help="Choose the template to edit.",
            )
            tmpl = templates[selected_template]

            instructions = st.text_area(
                "Summary instructions",
                value=tmpl.get("instructions", ""),
                placeholder="Summarize the problem, explain root cause, then list solution and lessons learned.",
                height=160,
                help="Describe the report format in plain language. The app converts it into structured sections.",
            )
            tone = st.selectbox(
                "Tone",
                ["simple", "detailed", "technical", "executive"],
                index=["simple", "detailed", "technical", "executive"].index(tmpl.get("tone", "simple")),
                help="Controls writing style for generated summaries.",
            )
            length = st.selectbox(
                "Length",
                ["short", "medium", "long"],
                index=["short", "medium", "long"].index(tmpl.get("length", "short")),
                help="Controls approximate detail level.",
            )

            if st.button("Save summary template", key="save_summary_template", help="Parse and save this summary configuration."):
                if not instructions.strip():
                    st.warning("Please enter summary instructions.")
                else:
                    try:
                        parsed = parse_summary_instructions(instructions)
                        templates[selected_template] = {
                            "sections": parsed.get("sections", []),
                            "tone": parsed.get("tone", tone),
                            "length": parsed.get("length", length),
                            "instructions": instructions,
                        }
                        save_templates(templates, user_id=user_id)
                        st.success("Template saved successfully.")
                        rerun()
                    except Exception:
                        st.error("Could not understand instructions. Please rephrase.")

        st.markdown("### Create new template")
        new_template_name = st.text_input("Template name", help="Create a reusable report format.")
        if st.button("Create template", help="Add a new summary template."):
            if not new_template_name:
                st.error("Template name required.")
            elif new_template_name in templates:
                st.error("Template already exists.")
            else:
                templates[new_template_name] = {
                    "sections": [],
                    "tone": "simple",
                    "length": "short",
                    "instructions": "",
                }
                save_templates(templates, user_id=user_id)
                st.success("Template created.")
                rerun()

    with system_tab:
        st.markdown("### Workspace overview")
        memories = mem_manager.list_memories()
        st.write(f"Saved memories: **{len(memories)}**")
        st.write(f"Search features: **{'Available' if emb_engine else 'Needs setup'}**")
        guidance("This page shows whether the workspace is ready for normal use.")

        if memories:
            st.markdown("### Memory access")
            selected_mem = st.selectbox(
                "Select memory",
                memories,
                key="workspace_memory_access",
                help="Review access, schema, and history for a memory.",
            )
            meta = mem_manager.get_memory_metadata(selected_mem)
            allowed = meta.get("allowed_user_ids", ["*"])
            access_label = "Shared workspace" if "*" in allowed else ", ".join(allowed)
            st.write(f"Access: **{access_label}**")

            access_mode = st.radio(
                "Update access",
                ["Shared workspace", "Only me", "Specific users"],
                horizontal=True,
                key="workspace_access_mode",
            )
            if access_mode == "Shared workspace":
                next_allowed = ["*"]
            elif access_mode == "Only me":
                next_allowed = [st.session_state["user"]["id"]]
            else:
                raw_allowed = st.text_input(
                    "Allowed user IDs",
                    value="" if "*" in allowed else ", ".join(allowed),
                    key="workspace_allowed_users",
                )
                next_allowed = sorted(set(
                    [item.strip() for item in raw_allowed.split(",") if item.strip()]
                    + [st.session_state["user"]["id"]]
                ))

            if st.button("Save access", help="Update who can view and add to this memory."):
                mem_manager.update_memory_permissions(selected_mem, next_allowed)
                st.success("Memory access updated.")
                st.rerun()

            with st.expander("Memory structure", expanded=False):
                schema = meta.get("schema") or {}
                if schema:
                    st.dataframe(
                        pd.DataFrame([
                            {"Column": col, "Type": details.get("type", ""), "Required": details.get("required", False)}
                            for col, details in schema.items()
                        ]),
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    st.info("No schema information stored yet. Save new records to refresh the structure.")

            with st.expander("History", expanded=False):
                audit_log = meta.get("audit_log") or []
                if audit_log:
                    st.dataframe(pd.DataFrame(audit_log), hide_index=True, width="stretch")
                else:
                    st.info("No history is available yet.")

