# NMR Streamlit App

A private Streamlit app for NMR analysis.

## Files
- `app.py` - main app
- `requirements.txt` - Python dependencies
- `.gitignore` - ignores local env and secrets

## Local run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

If no secret is configured, the app will still run locally without a password.
