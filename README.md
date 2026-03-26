# PLQY Streamlit App

A private Streamlit app for PLQY analysis.

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

## Optional password protection on Streamlit Cloud
In Streamlit app settings, add a secret:

```toml
app_password = "your-password-here"
```

If no secret is configured, the app will still run locally without a password.
