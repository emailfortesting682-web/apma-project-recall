
# apma_app.py — Stable + Append-Safe + Config-Aware + Column-Safe (NO FEATURE LOSS)

import streamlit as st
import os
import pandas as pd
from streamlit import rerun
from datetime import date

from modules.file_manager import MemoryManager
from modules.data_handler import DataHandler
from modules.embeddings_engine import EmbeddingsEngine
from modules.recall_engine import RecallEngine
from modules.utils import ensure_data_dirs
from modules import user_manager
from modules.manual_config import load_config, save_config
from modules.summary_templates import load_templates, save_templates
from modules.summary_parser import parse_summary_instructions
from modules.download_utils import export_csv, export_excel, export_pdf, export_word


# =====================================================
# CONFIGURATION
# =====================================================
try:
    OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
except:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

REQUIRED_COLS = [
    'COMMESSA', 'CLIENTE', 'ANNO', 'TIPO MACCHINA', 'APPLICAZIONE',
    'TIPO PROBLEMA', 'DESCRIZIONE', 'SOLUZIONE LESSON LEARNED',
    'DATA INSERIMENTO', 'RCPRD', 'REPORT CANTIERE',
    'CONCERNED DEPARTMENTS', 'REPORT RIUNIONE CHIUSURA PROGETTO'
]

SYSTEM_COLS = {
    "__semantic_text__", "AddedBy", "df_idx",
    "TextScore", "PhaseBonus", "CategoryBonus", "FinalScore"
}

# =====================================================
# HELPERS
# =====================================================
def normalize(col: str) -> str:
    return col.lower().replace(" ", "").replace("_", "")

def build_semantic_text(df: pd.DataFrame) -> pd.Series:
    cfg = load_config()
    semantic_cols = [
        col for col, meta in cfg.items()
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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="APMA — AI Project Memory Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

ensure_data_dirs()
st.title("🧠 AI Project Memory Assistant (APMA)")

# =====================================================
# AUTH
# =====================================================
if 'user' not in st.session_state:
    st.session_state['user'] = None

if st.session_state['user'] is None:
    st.sidebar.header("🔐 Login")
    auth_mode = st.sidebar.radio("Mode", ["Login", "Create account"])

    if auth_mode == "Create account":
        fn = st.sidebar.text_input("First name")
        ln = st.sidebar.text_input("Last name")
        uid = st.sidebar.text_input("ID Number")
        pw = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Create"):
            ok, msg = user_manager.create_user(fn, ln, uid, pw)
            st.sidebar.success(msg) if ok else st.sidebar.error(msg)
    else:
        uid = st.sidebar.text_input("ID Number")
        pw = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Login"):
            ok, msg, profile = user_manager.authenticate(uid, pw)
            if ok:
                st.session_state['user'] = profile
                rerun()
            else:
                st.sidebar.error(msg)

# =====================================================
# CORE OBJECTS
# =====================================================
mem_manager = MemoryManager(data_dir="data")

emb_engine = None
if OPENAI_API_KEY:
    try:
        emb_engine = EmbeddingsEngine()
    except:
        emb_engine = None

recall_engine = RecallEngine(
    emb_engine=emb_engine,
    mem_manager=mem_manager,
    category_col="TIPO MACCHINA",
    phase_col="APPLICAZIONE",
    problem_col="DESCRIZIONE",
    solution_col="SOLUZIONE LESSON LEARNED"
)

# =====================================================
# USER BAR
# =====================================================
if st.session_state.get("user"):
    u = st.session_state["user"]
    col1, col2 = st.columns([4, 1])
    col1.markdown(f"**Logged in:** {u['first_name']} {u['last_name']} ({u['id']})")
    if col2.button("Logout"):
        st.session_state['user'] = None
        rerun()

# =====================================================
# MODE SELECT
# =====================================================
mode = st.sidebar.selectbox("📂 Mode", [
    "Upload / Update Memory",
    "Query Knowledge Base",
    "Settings"
])

# =====================================================
# UPLOAD / UPDATE MEMORY
# =====================================================
if mode == "Upload / Update Memory":
    st.header("📤 Upload / Update Memory")

    if not st.session_state.get("user"):
        st.warning("Login required.")
        st.stop()

    uploaded = st.file_uploader("Upload CSV / Excel", ["csv", "xlsx"])

    if uploaded:
        df, err = DataHandler.read_and_validate(uploaded, required_cols=REQUIRED_COLS)
        if err:
            st.error(err)
        else:
            st.success(f"Validated {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)

            memories = mem_manager.list_memories()
            
            file_mem_mode = st.radio(
                "Save uploaded file to:",
                ["Create new memory", "Append to existing memory"],
                horizontal=True,
                key="file_mem_mode"
            )
            
            if file_mem_mode == "Create new memory":
                mem_name = st.text_input(
                    "New memory name",
                    key="file_new_memory_name"
                )
            else:
                if memories:
                    mem_name = st.selectbox(
                        "Select existing memory",
                        memories,
                        key="file_existing_memory"
                    )
                else:
                    st.warning("No existing memories available")
                    mem_name = None
            
            if st.button("Save file data"):
                if not mem_name:
                    st.error("Please select or enter a memory name.")
                    st.stop()
            
                df["AddedBy"] = st.session_state["user"]["id"]
                df["__semantic_text__"] = build_semantic_text(df)
            
                df = append_to_memory(mem_manager, mem_name, df)
                meta = mem_manager.create_or_update_memory(mem_name, df)
            
                if emb_engine:
                    emb_engine.index_dataframe(
                        meta["memory_path"],
                        df,
                        id_prefix=meta["memory_id"]
                    )
            
                st.success(
                    f"File data {'appended to' if file_mem_mode == 'Append to existing memory' else 'saved as new'} memory '{mem_name}'"
                )


    # ---------------- MANUAL ENTRY ----------------
    st.markdown("---")
    st.subheader("✍️ Manual Entry")

    config = load_config()
    manual_data = {}

    memories = mem_manager.list_memories()

    mem_mode = st.radio(
        "Save manual entry to:",
        ["Create new memory", "Append to existing memory"],
        horizontal=True
    )

    target_memory = None
    if mem_mode == "Create new memory":
        target_memory = st.text_input(
            "New memory name",
            key="manual_new_memory_name"
        )
    else:
        if memories:
            target_memory = st.selectbox(
                "Select memory",
                memories,
                key="manual_existing_memory"
            )
        else:
            st.warning("No existing memories available")
            target_memory = None


    with st.form("manual_dynamic"):
        cols = st.columns(4)
        col_idx = 0

        for field, meta in config.items():
            key = f"manual_{field}"
            with cols[col_idx]:
                if meta["type"] == "text":
                    manual_data[field] = st.text_input(field, key=key)
                elif meta["type"] == "select":
                    manual_data[field] = st.selectbox(field, meta.get("options", []), key=key)
                elif meta["type"] == "date":
                    if meta.get("mode") == "year":
                        y = st.selectbox(field, list(range(2000, date.today().year + 1)), key=key)
                        manual_data[field] = str(y)
                    else:
                        d = st.date_input(field, key=key)
                        manual_data[field] = d.isoformat()
            col_idx = (col_idx + 1) % 4

        submitted = st.form_submit_button("Add row")

    if submitted:
        st.session_state.setdefault("manual_rows", []).append(manual_data)
        st.success("Row added")

    if st.session_state.get("manual_rows"):
        st.markdown("### 📄 Pending Manual Entries")
        st.dataframe(pd.DataFrame(st.session_state["manual_rows"]), use_container_width=True)

        if st.button("💾 Save Manual Entries"):
            if not target_memory:
                st.error("Please select or create a memory.")
            else:
                df_manual = pd.DataFrame(st.session_state["manual_rows"])

                for col in REQUIRED_COLS:
                    if col not in df_manual.columns:
                        df_manual[col] = ""

                df_manual["AddedBy"] = st.session_state["user"]["id"]
                df_manual["__semantic_text__"] = build_semantic_text(df_manual)

                df_manual = append_to_memory(mem_manager, target_memory, df_manual)
                meta = mem_manager.create_or_update_memory(target_memory, df_manual)

                if emb_engine:
                    full_df = mem_manager.load_memory_dataframe(target_memory)
                    emb_engine.index_dataframe(
                        meta["memory_path"],
                        full_df,
                        id_prefix=meta["memory_id"]
                    )



                st.session_state["manual_rows"] = []
                st.success(f"Saved to memory '{target_memory}'")
                st.rerun()

# =====================================================
# QUERY MODE
# =====================================================
elif mode == "Query Knowledge Base":
    st.header("🔍 Query")
    templates = load_templates()
    if not templates:
        st.warning("No summary templates found. Please create one in Settings.")
        st.stop()

    
    summary_template_name = st.selectbox(
        "Select summary format",
        list(templates.keys()),
        key="query_summary_template"
    )
    
    selected_template = templates.get(summary_template_name)

    mems = mem_manager.list_memories()
    if not mems:
        st.warning("No memories")
        st.stop()

    mem = st.selectbox("Memory", mems)

    query_mode = st.radio(
        "Query type",
        ["AI Semantic Search", "Structured Filter Search"],
        horizontal=True
    )

    if query_mode == "Structured Filter Search":
        FILTERABLE_COLUMNS = {
            "COMMESSA": "COMMESSA",
            "CLIENTE": "CLIENTE",
            "ANNO": "ANNO",
            "TIPO MACCHINA": "TIPO MACCHINA",
            "APPLICAZIONE": "APPLICAZIONE",
            "TIPO PROBLEMA": "TIPO PROBLEMA"
        }

        c1, c2 = st.columns(2)
        col = c1.selectbox("Filter by", FILTERABLE_COLUMNS.keys())
        val = c2.text_input("Value")
        exact = st.checkbox("Exact match", False)

        if st.button("Filter"):
            df = recall_engine.filter_memory(mem, FILTERABLE_COLUMNS[col], val, exact)
            st.dataframe(df, use_container_width=True)
            st.info(f"{len(df)} records found")

    #
    else:
        q = st.text_area("Query")
    
        if st.button("Search"):
            if not emb_engine:
                st.error("Embeddings engine not available.")
                st.stop()
    
            if not q.strip():
                st.warning("Please enter a query.")
                st.stop()
    
            res = recall_engine.query_memory(mem, q)
    
            if res.empty:
                st.info("No matching results found.")
                st.stop()
    
            st.session_state["last_result_df"] = res
            st.session_state["last_query"] = q
    
            insights = recall_engine.generate_structured_insights(res)
    
            template = templates[summary_template_name]
            answer = recall_engine.generate_llm_summary(
                insights=insights,
                query=q,
                template=template,
                instructions=template.get("instructions", "")
            )
    
            st.session_state["last_summary"] = answer
    
    
    # ================= DISPLAY =================
    # ================= DISPLAY =================
    if "last_result_df" in st.session_state:
        df = st.session_state["last_result_df"]
        summary = st.session_state.get("last_summary", "")
    
        st.dataframe(df, use_container_width=True)
    
        st.markdown("### 🧠 Analysis Summary")
        st.markdown(summary)
    
        # ================= DOWNLOAD OPTIONS =================
        st.markdown("## ⬇️ Download Options")
    
        download_type = st.radio(
            "What do you want to download?",
            ["Only Table Results", "Only Summary Report", "Full Report (Table + Summary)"],
            horizontal=True
        )
    
        format_choice = st.selectbox(
            "Select format",
            ["CSV", "Excel", "PDF", "Word"],
            key="download_format_choice"
        )
    
        # ---------- EXPORT CONTENT ----------
        if download_type == "Only Table Results":
            export_df = df
            export_summary = ""
    
        elif download_type == "Only Summary Report":
            export_df = pd.DataFrame()
            export_summary = summary
    
        else:
            export_df = df
            export_summary = summary
    
        # ---------- DOWNLOAD BUTTON ----------
        if format_choice == "CSV":
            data = export_csv(export_df, export_summary)
            st.download_button("⬇️ Download CSV", data, "report.csv", "text/csv")
    
        elif format_choice == "Excel":
            data = export_excel(export_df, export_summary)
            st.download_button(
                "⬇️ Download Excel",
                data,
                "report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
        elif format_choice == "PDF":
            data = export_pdf(export_df, export_summary)
            st.download_button(
                "⬇️ Download PDF",
                data,
                "report.pdf",
                "application/pdf"
            )
    
        elif format_choice == "Word":
            data = export_word(export_df, export_summary)
            st.download_button(
                "⬇️ Download Word",
                data,
                "report.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )


# =====================================================
# SETTINGS
# =====================================================
if mode == "Settings":
    st.header("⚙️ Settings")

    st.subheader("🛠 Manual Entry Configuration")

    cfg = load_config()
    # Columns from memories (dataframes)
    memory_cols = get_existing_columns(mem_manager)
    
    # Columns from manual config
    config_cols = list(load_config().keys())
    
    # Merge both (case-insensitive, no duplicates)
    existing_cols = sorted({
        c for c in memory_cols + config_cols
        if c not in SYSTEM_COLS
    })

    existing_norm = {normalize(c): c for c in existing_cols}

    FIELD_ADD = "__ADD_NEW__"
    
    field = st.selectbox(
        "Select field",
        list(cfg.keys()) + [FIELD_ADD],
        key="settings_field_selector",
        format_func=lambda x: "➕ Add new" if x == FIELD_ADD else x
    )



    if field == FIELD_ADD:
        st.markdown("### Add Column")

        col_choice = st.selectbox(
            "Choose existing column (optional)",
            ["— None —"] + existing_cols
        )

        custom_name = st.text_input("Or enter new column name")
        new_type = st.selectbox("Type", ["text", "select", "date"])
        

        final_name = col_choice if col_choice != "— None —" else custom_name.strip()
        
        if not final_name:
            st.info("⬆️ Select or enter a column name to enable saving.")
            
        
        # Case 1: user selected from dropdown → ALWAYS allowed
       # if col_choice != "— None —":
         #   pass
        selected_existing_col = None
        if col_choice != "— None —":
            selected_existing_col = col_choice


        if selected_existing_col:
            st.markdown("### Existing Column Actions")
    
            rename_to = st.text_input(
                "Rename selected column",
                value=selected_existing_col,
                key="rename_existing_column"
            )
    
            col_r1, col_r2 = st.columns(2)
    
            with col_r1:
                if st.button("✏️ Rename Column"):
                    if normalize(rename_to) in map(normalize, cfg.keys()):
                        st.error("Column already exists in manual configuration.")
                        st.stop()
    
                    old_meta = cfg.get(selected_existing_col, {})
                    cfg[rename_to] = old_meta

    
                    if selected_existing_col in cfg:
                        del cfg[selected_existing_col]
    
                    save_config(cfg)
                    st.success("Column renamed successfully")
                    st.rerun()
    
            with col_r2:
                if st.button("🗑 Remove Column"):
                    if selected_existing_col in REQUIRED_COLS:
                        st.error("This column is required and cannot be removed.")
                        st.stop()
    
                    if selected_existing_col in cfg:
                        del cfg[selected_existing_col]
    
                    save_config(cfg)
                    st.warning("Column removed from manual entry configuration")
                    st.rerun()













        # Case 2: user typed a name → check duplicates (case-insensitive)
        if st.button("💾 Save column"):
        
            if not final_name:
                st.error("Please select or enter a column name.")
            elif (
                col_choice == "— None —"
                and (
                    normalize(final_name) in existing_norm
                    or normalize(final_name) in map(normalize, cfg.keys())
                )
            ):
                st.warning("⚠️ Column already exists. Please choose from the list.")
            else:
                cfg[final_name] = {"type": new_type}
                save_config(cfg)
        
                st.session_state.pop("settings_field_selector", None)
                st.success(f"Column '{final_name}' added")
                st.rerun()

            


    else:
        if field not in cfg:
            st.warning("Invalid field selected. Please reselect.")
            st.stop()
        meta = cfg[field]
        new_field_name = st.text_input("Rename field", value=field)

        meta["type"] = st.selectbox(
            "Field type",
            ["text", "select", "date"],
            index=["text", "select", "date"].index(meta["type"])
        )

        if meta["type"] == "select":
            opts = st.text_area(
                "Dropdown options (one per line)",
                "\n".join(meta.get("options", []))
            )
            meta["options"] = [o.strip() for o in opts.splitlines() if o.strip()]

        if meta["type"] == "date":
            meta["mode"] = st.radio("Date mode", ["full", "year"])

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("💾 Save changes"):
                new_norm = normalize(new_field_name)
                existing_norms = {normalize(c): c for c in cfg.keys() if c != field}
        
                if new_norm in existing_norms:
                    st.error("A column with this name already exists.")
                    st.stop()
        
                cfg[new_field_name] = meta
                if new_field_name != field:
                    del cfg[field]
        
                save_config(cfg)
                st.success("Field updated")
                st.rerun()


        with col_b:
            if st.button("🗑 Delete field"):
                if field in REQUIRED_COLS:
                    st.error("This is a required system column and cannot be deleted.")
                    st.stop()
        
                save_config({k: v for k, v in cfg.items() if k != field})
                st.warning(f"Field '{field}' deleted")
                st.rerun()

    st.subheader("🧠 Summary Output Configuration")
    
    templates = load_templates()
    template_names = list(templates.keys())
    
    if not template_names:
        st.warning("No summary templates found. Create one below ⬇️")
    else:
        #
        
    
        selected_template = st.selectbox(
            "Select summary template",
            template_names,
            key="selected_summary_template"
        )
    selected_template = st.session_state.get("selected_summary_template")
    
    if selected_template:
        

        tmpl = templates[selected_template]
    
        instructions = st.text_area(
            "📝 Summary instructions (Natural language)",
            value=tmpl.get("instructions", ""),
            placeholder=(
                "Example:\n"
                "Summarize the problem briefly, explain the root cause, "
                "then list the solution and lessons learned."
            ),
            height=160
        )
    
    
        tone = st.selectbox(
            "Tone",
            ["simple", "detailed", "technical", "executive"],
            index=["simple", "detailed", "technical", "executive"].index(
                tmpl.get("tone", "simple")
            )
        )
    
        length = st.selectbox(
            "Length",
            ["short", "medium", "long"],
            index=["short", "medium", "long"].index(
                tmpl.get("length", "short")
            )
        )
    
    
    
            #
        if st.button("💾 Save Summary Template", key="save_summary_template"):
            
            if not instructions.strip():
                st.warning("Please enter summary instructions.")
            else:
        
                try:
                    parsed = parse_summary_instructions(instructions)
            
                    templates[selected_template] = {
                        "sections": parsed.get("sections", []),
                        "tone": parsed.get("tone", tone),
                        "length": parsed.get("length", length),
                        "instructions": instructions
                    }
            
                    save_templates(templates)
                    st.success("Template saved successfully ✅")
                    rerun()
            
                except Exception:
                    st.error("Could not understand instructions. Please rephrase.")
        
        
                #save_templates(templates)
 
    
    st.markdown("---")
    
    new_template_name = st.text_input("Create new summary template")
    
    if st.button("➕ Create Template"):
        if not new_template_name:
            st.error("Template name required")
        elif new_template_name in templates:
            st.error("Template already exists")
        else:
            
            templates[new_template_name] = {
                "sections": [],
                "tone": "simple",
                "length": "short",
                "instructions": ""
            }
    
            save_templates(templates)
            st.success("Template created")
            rerun()   # ✅


