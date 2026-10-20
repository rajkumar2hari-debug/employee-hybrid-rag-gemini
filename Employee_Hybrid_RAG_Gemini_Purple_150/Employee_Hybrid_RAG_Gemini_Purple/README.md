# Employee Hybrid RAG Assistant — Gemini Cloud

A Streamlit demo with a purple interface, 150 synthetic employee records, keyword-based retrieval, and Gemini Cloud answers grounded in retrieved CSV rows.

## Files

- `app.py` — Streamlit interface, retrieval logic, Gemini integration
- `data/employees.csv` — 150 synthetic demo employees
- `requirements.txt` — Python dependencies
- `.streamlit/secrets.toml.example` — example only; do not put a real key here in a public repository

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Set your API key as an environment variable (recommended):

   PowerShell:
   ```powershell
   $env:GEMINI_API_KEY="PASTE_YOUR_KEY_HERE"
   streamlit run app.py
   ```

   Replace the placeholder locally; never commit the key.

## Deploy on Streamlit Community Cloud

1. Upload this project folder's contents to a new GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and set the main file path to `app.py`.
3. Open the app's **Settings → Secrets** and add:

   ```toml
   GEMINI_API_KEY = "paste-your-real-key-here"
   ```

4. Save and reboot the app.

Do not commit real API keys, `.env` files, or real employee data. The included records are synthetic demonstration data.

## Model

The configured Gemini model is `gemini-3.8-flash`. If your Google AI Studio account does not have access to this model, select a model currently available to your account and update `MODEL_NAME` in `app.py`.

## Important design note

This starter uses lightweight keyword-based retrieval over CSV rows, then supplies retrieved rows to Gemini for questions that are not handled as direct employee-field lookups. It is a beginner-friendly retrieval-grounded prototype, not a vector-database implementation. Review answers against the source records before using it for real HR decisions.
