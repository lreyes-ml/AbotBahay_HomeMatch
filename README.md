# HomeMatch AI Streamlit App (v4)

AbotBahay-style Streamlit interface with 2 operational views:
- Applicant Pipeline
- Document Drafter

The app uses `homematch_model.pkl` if available. If missing, it automatically falls back to a deterministic readiness scorer.

## Project Files
- `app.py`: Main Streamlit app
- `.streamlit/config.toml`: Theme config
- `.streamlit/secrets.toml.example`: Secrets template
- `scripts/smoke_test.sh`: Pre-deploy smoke checks

## Local Run
```bash
cd homematch_streamlit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and set ANTHROPIC_API_KEY
./scripts/smoke_test.sh
streamlit run app.py
```

## GitHub + Streamlit Cloud Deploy
1. Push repository to GitHub.
2. In Streamlit Cloud, create app from repo.
3. Set entrypoint to `homematch_streamlit/app.py`.
4. Add secret in Streamlit Cloud:
```toml
ANTHROPIC_API_KEY = "your_key_here"
```
5. Deploy and verify pages load:
- Applicant Pipeline shows KPI cards and matrix.
- Document Drafter generates draft output in English, Filipino, Kapampangan, or Bisaya.
- Case manager in the header uses a dropdown selector with 4 options:
  - `Reynan Tayag` (default)
  - `Leo Reyes`
  - `Ruth Nartatez`
  - `Audrey Yaneza`

## Top Controls Use Cases
- Case Manager selector:
  - Injects the selected manager name into drafting prompts and signature context.
  - Supports explicit ownership assignment before drafting.
- Draft Mode selector (inside Document Drafter):
  - Chooses output language/dialect: English, Filipino, Kapampangan, Bisaya.
- Copy Text:
  - Uses browser clipboard API to copy the current draft letter content.
  - Remains visible with explicit styling (no hidden text on non-hover state).

## Deployment Checklist
- [ ] `./scripts/smoke_test.sh` passes.
- [ ] `app.py` syntax check passes.
- [ ] `ANTHROPIC_API_KEY` added in Streamlit Cloud secrets.
- [ ] No real applicant PII in demo records.
- [ ] Case-manager disclaimer visible in outputs.
- [ ] Model file strategy decided:
  - [ ] Commit `homematch_model.pkl` to repo, or
  - [ ] Use deterministic fallback only, or
  - [ ] Load model from remote storage.

## Notes
- LLM call path includes retry and model fallback chain.
- This is a decision-support tool; final decisions require case manager validation.
