import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai

APP_DIR = Path(__file__).resolve().parent
CSV_PATH = APP_DIR / "data" / "employees.csv"
MODEL_NAME = "gemini-3.8-flash"

st.set_page_config(page_title="Employee Hybrid RAG", page_icon="💜", layout="wide")

st.markdown("""
<style>
.stApp {
  background: radial-gradient(circle at top left, #5b21b6 0, #251044 35%, #100b20 78%);
  color: #f8f5ff;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.block-container { padding-top: 2rem; max-width: 1180px; }
.hero {
  padding: 1.8rem 2rem; border-radius: 24px;
  background: linear-gradient(120deg, rgba(139,92,246,.35), rgba(49,26,88,.65));
  border: 1px solid rgba(216,180,254,.28); margin-bottom: 1rem;
}
.hero h1 { color: #fff; margin-bottom: .35rem; }
.hero p { color: #e9d5ff; font-size: 1.05rem; }
div[data-testid="stMetric"] {
  background: rgba(255,255,255,.07); border: 1px solid rgba(216,180,254,.2);
  border-radius: 16px; padding: 1rem;
}
.stTextInput input, .stTextArea textarea {
  background: #1d1430 !important; color: #fff !important;
  border: 1px solid #8b5cf6 !important; border-radius: 12px !important;
}
.stButton button {
  background: linear-gradient(90deg, #9333ea, #7c3aed); color: white;
  border: 0; border-radius: 12px; padding: .55rem 1.2rem; font-weight: 600;
}
div[data-testid="stExpander"] { background: rgba(255,255,255,.05); border-radius: 14px; }
.small-note { color: #d8b4fe; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>💜 Employee Hybrid RAG Assistant</h1>
  <p>Ask questions about employee records using retrieval-grounded Gemini Cloud answers.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Employee CSV was not found at: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    required = {"employee_id","name","department","designation","annual_salary_inr","joining_date","office_location","employment_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))
    return df

try:
    employees = load_data()
except Exception as e:
    st.error(f"Could not load employee data: {e}")
    st.stop()

c1,c2,c3 = st.columns(3)
c1.metric("Employee records", f"{len(employees):,}")
c2.metric("Departments", str(employees["department"].nunique()))
c3.metric("Office locations", str(employees["office_location"].nunique()))

with st.expander("📁 View employee dataset"):
    st.dataframe(employees, use_container_width=True, hide_index=True)

def retrieve_records(question: str, df: pd.DataFrame, limit: int = 12):
    q = question.lower()
    # Exact employee ID lookup has priority
    ids = re.findall(r"\bEMP\d{4,}\b", question.upper())
    if ids:
        exact = df[df["employee_id"].isin(ids)]
        if not exact.empty:
            return exact.head(limit)
    # Match tokens against all fields; score rows by number of matching terms
    stop_words = {"what","when","where","which","who","how","many","much","does","did","the","is","are","of","for","in","on","a","an","and","list","show","tell","me","please","employee","employees","work","working","with","have","has","their","all","find","get"}
    tokens = [t for t in re.findall(r"[a-z0-9]+", q) if len(t) > 2 and t not in stop_words]
    searchable = df.astype(str).apply(lambda col: col.str.lower())
    scores = pd.Series(0, index=df.index, dtype=int)
    for token in tokens:
        for col in searchable.columns:
            scores += searchable[col].str.contains(re.escape(token), regex=True, na=False).astype(int)
    matches = df.loc[scores[scores > 0].sort_values(ascending=False).head(limit).index]
    return matches

def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GEMINI_API_KEY", "")

st.markdown("### 💬 Ask about employees")
question = st.text_input("Your question", placeholder="Example: What is the department and salary of EMP1004?")
ask = st.button("Ask Employee Assistant", type="primary")

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()
    retrieved = retrieve_records(question, employees)
    if retrieved.empty:
        st.warning("I couldn't find matching employee records in the dataset. Try an employee ID, department, or office location.")
        st.stop()

    st.markdown("### Answer")
    # Deterministic answers for common structured questions
    ql = question.lower()
    answer = None
    ids = re.findall(r"\bEMP\d{4,}\b", question.upper())
    selected = employees[employees["employee_id"].isin(ids)] if ids else pd.DataFrame()
    if not selected.empty and any(word in ql for word in ["department","salary","designation","joined","joining date","office","location","status","name"]):
        row = selected.iloc[0]
        facts=[]
        if "department" in ql: facts.append(f"Department: {row['department']}")
        if "salary" in ql or "annual salary" in ql: facts.append(f"Annual salary: ₹{int(row['annual_salary_inr']):,} INR")
        if "designation" in ql or "role" in ql: facts.append(f"Designation: {row['designation']}")
        if "join" in ql or "joining date" in ql: facts.append(f"Joining date: {row['joining_date']}")
        if "office" in ql or "location" in ql or "city" in ql: facts.append(f"Office location: {row['office_location']}")
        if "status" in ql or "active" in ql: facts.append(f"Employment status: {row['employment_status']}")
        if "name" in ql: facts.append(f"Name: {row['name']}")
        answer = "\n\n".join(facts)

    if answer:
        st.success(answer)
        st.caption("Source: exact employee record from the local CSV.")
    else:
        api_key = get_api_key()
        if not api_key:
            st.error("Gemini API key is not configured. Add GEMINI_API_KEY in Streamlit → App settings → Secrets. Do not put the key in app.py or GitHub.")
            st.stop()
        context = retrieved.to_csv(index=False)
        prompt = f"""You are an employee-data assistant. Answer only from the CSV records provided below.
If the records do not contain enough information, say you cannot determine the answer from the available records.
Do not invent employees, fields, salaries, dates, counts, or conclusions.
User question: {question}

Retrieved CSV records:
{context}

Give a concise answer and mention the relevant employee IDs when possible."""
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            st.write(response.text or "The model returned an empty answer.")
        except Exception as e:
            st.error("Gemini request failed. Check the model availability, API key, and quota.")
            st.code(str(e))
            st.info("Temporary service errors can happen. Wait briefly and retry; the app and CSV are still available.")
    with st.expander("🔎 Retrieved source records", expanded=True):
        st.dataframe(retrieved, use_container_width=True, hide_index=True)

st.markdown('<p class="small-note">Demo dataset: 150 synthetic employee records. Do not use this sample data as real HR information.</p>', unsafe_allow_html=True)
