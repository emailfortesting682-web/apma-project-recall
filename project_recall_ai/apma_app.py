import os
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
    "df_idx",
    "TextScore",
    "PhaseBonus",
    "CategoryBonus",
    "FinalScore",
}


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
            --apma-navy: #172033;
            --apma-blue: #2563eb;
            --apma-green: #0f766e;
            --apma-border: #d8dee8;
            --apma-muted: #667085;
            --apma-bg: #f7f9fc;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }
        h1, h2, h3 {
            color: var(--apma-navy);
            letter-spacing: 0;
        }
        [data-testid="stSidebar"] {
            background: #f3f6fb;
            border-right: 1px solid var(--apma-border);
        }
        .apma-hero {
            border: 1px solid var(--apma-border);
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border-radius: 8px;
            padding: 22px 24px;
            margin-bottom: 18px;
        }
        .apma-title {
            margin: 0;
            color: var(--apma-navy);
            font-size: 30px;
            line-height: 1.2;
            font-weight: 760;
        }
        .apma-subtitle {
            margin: 8px 0 0 0;
            color: var(--apma-muted);
            font-size: 15px;
            line-height: 1.5;
        }
        .apma-help {
            border-left: 4px solid var(--apma-blue);
            background: #eef5ff;
            padding: 12px 14px;
            color: #1d2939;
            margin: 8px 0 18px 0;
            border-radius: 4px;
        }
        .apma-card {
            border: 1px solid var(--apma-border);
            border-radius: 8px;
            background: #ffffff;
            padding: 16px 18px;
            min-height: 112px;
        }
        .apma-card-title {
            color: var(--apma-muted);
            font-size: 13px;
            font-weight: 650;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .apma-card-value {
            color: var(--apma-navy);
            font-size: 28px;
            font-weight: 760;
        }
        .apma-workflow {
            border: 1px solid var(--apma-border);
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            margin-bottom: 10px;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 6px;
            font-weight: 650;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, description: str):
    st.markdown(
        f"""
        <div class="apma-hero">
            <div class="apma-title">{title}</div>
            <p class="apma-subtitle">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def guidance(text: str):
    st.markdown(f'<div class="apma-help">{text}</div>', unsafe_allow_html=True)


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


def get_secret_value(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


def is_admin_user() -> bool:
    user = st.session_state.get("user")
    if not user:
        return False
    raw_admin_ids = get_secret_value("APMA_ADMIN_IDS", "")
    if not raw_admin_ids.strip():
        return True
    admin_ids = {item.strip() for item in raw_admin_ids.split(",") if item.strip()}
    return user.get("id") in admin_ids


def require_admin():
    require_login()
    if not is_admin_user():
        st.error("Settings are restricted to administrators.")
        st.info("Ask the app owner to add your ID to APMA_ADMIN_IDS in Streamlit secrets.")
        st.stop()


def memory_record_count(mem_manager, memory_id: str) -> int:
    try:
        return len(mem_manager.load_memory_dataframe(memory_id))
    except Exception:
        return 0


def save_memory_with_embeddings(mem_manager, emb_engine, name: str, df: pd.DataFrame):
    meta = mem_manager.create_or_update_memory(name, df)
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


def clear_search_state():
    for key in ["last_result_df", "last_summary", "last_query", "last_result_memory"]:
        st.session_state.pop(key, None)


def go_to(page: str):
    st.session_state["workspace_nav"] = page
    rerun()


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
        ["CSV", "Excel", "PDF", "Word"],
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


st.set_page_config(
    page_title="APMA - AI Project Memory Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_data_dirs()
inject_professional_theme()

if "user" not in st.session_state:
    st.session_state["user"] = None

mem_manager = MemoryManager(data_dir="data")

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
st.sidebar.markdown("## APMA")
st.sidebar.caption("AI Project Memory Assistant")
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
    page_header(
        "Project Memory Dashboard",
        "A clear starting point for reviewing system status, saved memories, and the next recommended actions.",
    )

    memories = mem_manager.list_memories()
    total_records = sum(memory_record_count(mem_manager, mem) for mem in memories)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Saved memories", str(len(memories)), "Knowledge bases available for search")
    with c2:
        metric_card("Stored records", str(total_records), "Rows available across all memories")
    with c3:
        metric_card("AI search", "Ready" if emb_engine else "Needs key", "OpenAI embeddings status")
    with c4:
        storage = "Supabase" if getattr(mem_manager, "use_supabase", False) else "Local"
        metric_card("Storage", storage, "Current persistence backend")

    st.markdown("### Recommended workflow")
    steps = [
        ("1. Add project data", "Use Data Upload for CSV/Excel files or Manual Entry for individual lessons learned."),
        ("2. Validate and save memory", "Create a new memory or append records to an existing memory."),
        ("3. Search and summarize", "Use Search & Insights to retrieve similar historical cases and generate a report."),
        ("4. Configure when needed", "Use Settings to adjust manual fields and summary templates."),
    ]
    for title, body in steps:
        st.markdown(f'<div class="apma-workflow"><strong>{title}</strong><br>{body}</div>', unsafe_allow_html=True)

    st.markdown("### Quick actions")
    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        if st.button("Upload project file", help="Go to the bulk CSV/Excel upload workflow."):
            go_to("Data Upload")
    with qa2:
        if st.button("Add manual record", help="Go to the manual lesson-learned entry workflow."):
            go_to("Manual Entry")
    with qa3:
        if st.button("Search memories", help="Go to semantic search and reporting."):
            go_to("Search & Insights")

    st.markdown("### Existing memories")
    if memories:
        rows = [{"Memory": mem, "Records": memory_record_count(mem_manager, mem)} for mem in memories]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        if st.session_state.get("user"):
            guidance("No memories found yet. Use Upload project file or Add manual record to create the first project memory.")
        else:
            guidance("Log in from the sidebar to create and search project memories.")


elif mode == "Data Upload":
    page_header(
        "Data Upload",
        "Import CSV or Excel project records, validate the required schema, and save them into a searchable memory.",
    )
    require_login()
    guidance("Use this page for bulk import. The file must contain the required project columns before it can be saved.")

    template_df = sample_template_dataframe()
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download CSV template",
            template_df.to_csv(index=False).encode("utf-8"),
            "apma_upload_template.csv",
            "text/csv",
            help="Download a blank CSV with the exact columns required by APMA.",
        )
    with dl2:
        st.download_button(
            "Download Excel template",
            excel_bytes(template_df),
            "apma_upload_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download a blank Excel template with the exact columns required by APMA.",
        )

    with st.expander("Required upload columns", expanded=False):
        st.write("Your CSV or Excel file must include these columns:")
        st.dataframe(pd.DataFrame({"Required column": REQUIRED_COLS}), hide_index=True, use_container_width=True)

    uploaded = st.file_uploader(
        "Upload CSV or Excel file",
        ["csv", "xlsx"],
        help="Accepted formats are .csv and .xlsx. The app validates required columns before saving.",
    )

    if uploaded:
        df, err = DataHandler.read_and_validate(uploaded, required_cols=REQUIRED_COLS)
        if err:
            st.error(err)
            st.info("Check the source file headers and try uploading again.")
        else:
            st.success(f"Validated {len(df)} rows.")
            st.dataframe(df.head(20), use_container_width=True)

            memories = mem_manager.list_memories()
            file_mem_mode = st.radio(
                "Save uploaded file to",
                ["Create new memory", "Append to existing memory"],
                horizontal=True,
                key="file_mem_mode",
                help="Create a separate knowledge base or add these rows to one that already exists.",
            )

            if file_mem_mode == "Create new memory":
                mem_name = st.text_input(
                    "New memory name",
                    key="file_new_memory_name",
                    help="Use a clear business name, for example Client-2026-Lessons or Packaging-Line-Issues.",
                )
            else:
                if memories:
                    mem_name = st.selectbox(
                        "Select existing memory",
                        memories,
                        key="file_existing_memory",
                        help="The uploaded rows will be appended and the search index rebuilt.",
                    )
                else:
                    st.warning("No existing memories are available yet.")
                    mem_name = None

            if st.button("Save file data", help="Store the validated file records and rebuild AI embeddings."):
                if not mem_name:
                    st.error("Please select or enter a memory name.")
                    st.stop()

                df["AddedBy"] = st.session_state["user"]["id"]
                df["__semantic_text__"] = build_semantic_text(df)
                final_df = append_to_memory(mem_manager, mem_name, df)
                save_memory_with_embeddings(mem_manager, emb_engine, mem_name, final_df)
                st.success(f"File data saved to memory '{mem_name}'.")


elif mode == "Manual Entry":
    page_header(
        "Manual Entry",
        "Capture individual project lessons learned when a complete upload file is not available.",
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
        st.markdown("#### Project information")
        cols = st.columns(3)
        for idx, field in enumerate([f for f in project_fields if f in config]):
            with cols[idx % 3]:
                render_manual_field(field)

        st.markdown("#### Problem and lesson learned")
        for field in [f for f in detail_fields if f in config]:
            render_manual_field(field)

        st.markdown("#### Reports and ownership")
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
        st.dataframe(pending_df, use_container_width=True)

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

                    df_manual["AddedBy"] = st.session_state["user"]["id"]
                    df_manual["__semantic_text__"] = build_semantic_text(df_manual)
                    final_df = append_to_memory(mem_manager, target_memory, df_manual)
                    save_memory_with_embeddings(mem_manager, emb_engine, target_memory, final_df)
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
        "Search historical project memories, review matching records, generate an AI summary, and export reports.",
    )
    require_login()
    guidance("Use semantic search for natural-language questions. Use structured filters when you know the exact field to inspect.")

    templates = load_templates()
    mems = mem_manager.list_memories()

    if not mems:
        st.warning("No memories found. Add records from Data Upload or Manual Entry first.")
        st.stop()

    top1, top2 = st.columns(2)
    with top1:
        mem = st.selectbox("Memory", mems, help="Select the project memory to search.")
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

    search_tab, filter_tab = st.tabs(["AI Semantic Search", "Structured Filter Search"])

    with search_tab:
        q = st.text_area(
            "Question or problem description",
            placeholder="Example: installation delays caused by unclear layout requirements",
            help="Describe the issue in natural language. The app finds similar historical records.",
        )

        if st.button("Search memory", help="Run AI semantic search and generate a summary."):
            if not emb_engine:
                st.error("Embeddings engine is not available. Check OPENAI_API_KEY in Streamlit secrets.")
                st.stop()
            if not q.strip():
                st.warning("Please enter a query.")
                st.stop()

            try:
                res = recall_engine.query_memory(mem, q)
            except FileNotFoundError:
                st.error("Embeddings were not found for this memory. Re-save or re-upload the memory to rebuild them.")
                st.stop()

            if res.empty:
                st.info("No matching results found.")
                st.stop()

            insights = recall_engine.generate_structured_insights(res)
            template = templates[summary_template_name]
            answer = recall_engine.generate_llm_summary(
                insights=insights,
                query=q,
                template=template,
                instructions=template.get("instructions", ""),
            )

            st.session_state["last_result_df"] = res
            st.session_state["last_summary"] = answer
            st.session_state["last_query"] = q
            st.session_state["last_result_memory"] = mem

    with filter_tab:
        filterable_columns = {
            "COMMESSA": "COMMESSA",
            "CLIENTE": "CLIENTE",
            "ANNO": "ANNO",
            "TIPO MACCHINA": "TIPO MACCHINA",
            "APPLICAZIONE": "APPLICAZIONE",
            "TIPO PROBLEMA": "TIPO PROBLEMA",
        }
        c1, c2, c3 = st.columns([1, 2, 1])
        col = c1.selectbox("Filter by", filterable_columns.keys(), help="Choose the column to search within.")
        val = c2.text_input("Value", help="Enter the value or partial text to match.")
        exact = c3.checkbox("Exact match", False, help="Require an exact value instead of partial matching.")

        if st.button("Apply filter", help="Return records that match the selected structured filter."):
            df = recall_engine.filter_memory(mem, filterable_columns[col], val, exact)
            st.session_state["last_result_df"] = df
            st.session_state["last_summary"] = ""
            st.session_state["last_result_memory"] = mem
            st.info(f"{len(df)} records found.")

    if "last_result_df" in st.session_state:
        df = st.session_state["last_result_df"]
        summary = st.session_state.get("last_summary", "")

        st.markdown("### Results")
        st.dataframe(df, use_container_width=True)

        if summary:
            st.markdown("### Analysis summary")
            st.markdown(summary)

        render_download_panel(df, summary)


elif mode == "Settings":
    page_header(
        "Settings",
        "Configure manual-entry fields, summary templates, and deployment readiness from one administration area.",
    )
    require_admin()

    if not get_secret_value("APMA_ADMIN_IDS", "").strip():
        st.info("Admin ID restriction is not configured yet. Add APMA_ADMIN_IDS to Streamlit secrets to limit Settings access.")

    fields_tab, templates_tab, system_tab = st.tabs(["Manual fields", "Summary templates", "System status"])

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
        templates = load_templates()
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
                        save_templates(templates)
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
                save_templates(templates)
                st.success("Template created.")
                rerun()

    with system_tab:
        st.markdown("### Deployment readiness")
        storage = "Supabase" if getattr(mem_manager, "use_supabase", False) else "Local data folder"
        st.write(f"Storage backend: **{storage}**")
        st.write(f"OpenAI embeddings: **{'Available' if emb_engine else 'Not available'}**")
        st.write(f"Saved memories: **{len(mem_manager.list_memories())}**")
        guidance("For client testing, storage should show Supabase and OpenAI embeddings should show Available.")
