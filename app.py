import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import anthropic
except Exception:  # pragma: no cover
    anthropic = None

st.set_page_config(page_title="AbotBahay HomeMatch AI", page_icon="HF", layout="wide")

st.markdown(
    """
<style>
:root {
  --bg: #f4f6fa;
  --card: #ffffff;
  --ink: #0f172a;
  --muted: #64748b;
  --accent: #f59e0b;
  --accent-dark: #d97706;
  --ok: #0f9f6e;
  --warn: #cc7a00;
  --risk: #d7263d;
  --line: #dde4ef;
  --panel: #080d17;
}

html, body, [class*="css"] {
  font-family: "Segoe UI", "Inter", sans-serif;
}

.stApp {
  background: var(--bg);
  color: var(--ink);
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #05080f 0%, #090f1c 100%);
  border-right: 1px solid #1a253a;
}

section[data-testid="stSidebar"] * {
  color: #d7e0ef;
}

.sidebar-logo {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 800;
  font-size: 1.85rem;
  letter-spacing: 0.01em;
}

.sidebar-sub {
  color: #8da0bf;
  font-size: 0.8rem;
  margin-top: -4px;
  margin-bottom: 14px;
}

.system-card {
  margin-top: 12px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 14px;
  padding: 12px;
}

.kpi {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}

.kpi-label {
  color: #8aa0bf;
  font-weight: 700;
  letter-spacing: 0.04em;
  font-size: 0.85rem;
  text-transform: uppercase;
}

.kpi-value {
  font-size: 2.15rem;
  font-weight: 800;
  margin-top: 4px;
}

.panel {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 18px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}

.panel-head {
  padding: 16px 18px;
  border-bottom: 1px solid var(--line);
}

.panel-body {
  padding: 16px 18px;
}

.dark-context {
  background: linear-gradient(160deg, #05070e 0%, #0b0f1b 100%);
  border: 1px solid #1d2940;
  border-radius: 24px;
  padding: 18px;
  color: #f5f9ff;
  margin-bottom: 16px;
}

.lang-chip {
  display: inline-block;
  background: rgba(245, 158, 11, 0.15);
  color: #fbbf24;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.82rem;
  font-weight: 700;
}

.risk-badge {
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 700;
  display: inline-block;
}

.risk-preferred {
  background: #e7f6ef;
  color: #0f9f6e;
}

.risk-standard {
  background: #fff5e6;
  color: #c66a00;
}

.risk-needs {
  background: #fdecef;
  color: #cc243c;
}

.applicant-card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 18px;
  height: 100%;
}

.avatar {
  width: 90px;
  height: 90px;
  border-radius: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 2rem;
  color: #c96a00;
  background: #fbf4e8;
  border: 1px solid #f2deae;
  margin: 8px auto 14px auto;
}

.msg-user {
  background: #f59e0b;
  color: #111827;
  padding: 10px 12px;
  border-radius: 16px 16px 4px 16px;
}

.msg-assistant {
  background: #ffffff;
  border: 1px solid var(--line);
  padding: 12px;
  border-radius: 16px 16px 16px 4px;
}

.meta-strip {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #ffffff;
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 12px 14px;
  margin-bottom: 14px;
}

.top-title {
  font-size: 2rem;
  font-weight: 800;
  margin: 0;
}

.top-sub {
  color: #60738f;
  margin-top: -2px;
}
</style>
""",
    unsafe_allow_html=True,
)

EMP_RISK_MAP = {
    "regular": 0.70,
    "contractual": 0.40,
    "self_employed": 0.50,
    "ofw": 0.60,
    "informal": 0.20,
}

DEFAULT_FEATURES = [
    "monthly_income",
    "employment_years",
    "dependents",
    "age",
    "pagibig_years",
    "has_informal_debt",
    "doc_completeness",
    "emp_type_score",
]

REQUIRED_DOCS = [
    "Government-issued ID",
    "PSA Birth Certificate",
    "Marriage/Death Certificate (if applicable)",
    "Income document (ITR / Pay Slips x3 / Business Permit)",
    "Pag-IBIG contribution printout (within 3 months)",
    "Barangay certificate of residency",
    "Proof of billing address",
]

APPLICANTS: List[Dict] = [
    {
        "id": "HELP-001",
        "name": "Maria Clara Santos",
        "role": "Public School Teacher",
        "age": 34,
        "civil_status": "Married",
        "dependents": 2,
        "monthly_income": 26000,
        "employment_status": "Employed (Regular)",
        "employment_type_code": "regular",
        "employment_tenure_years": 6,
        "pagibig_years": 5,
        "pagibig_status": "Active - 5 years contributions",
        "existing_loans": "None",
        "has_informal_debt": 0,
        "language": "Filipino",
        "preferred_location": "Bulacan",
        "submitted_docs": ["Government ID", "PSA Birth Certificate", "Pay Slip x3", "Pag-IBIG printout", "Barangay certificate"],
    },
    {
        "id": "HELP-002",
        "name": "Roberto Garcia",
        "role": "Tricycle Operator",
        "age": 45,
        "civil_status": "Married",
        "dependents": 3,
        "monthly_income": 14500,
        "employment_status": "Self-employed (tricycle)",
        "employment_type_code": "self_employed",
        "employment_tenure_years": 7,
        "pagibig_years": 1,
        "pagibig_status": "Inactive - needs contribution recovery",
        "existing_loans": "Informal lending (5-6)",
        "has_informal_debt": 1,
        "language": "Kapampangan",
        "preferred_location": "Pampanga",
        "submitted_docs": ["Barangay certificate", "Government ID"],
    },
    {
        "id": "HELP-003",
        "name": "Elena Pangilinan",
        "role": "BPO Professional",
        "age": 30,
        "civil_status": "Single",
        "dependents": 1,
        "monthly_income": 32000,
        "employment_status": "Employed (Regular)",
        "employment_type_code": "regular",
        "employment_tenure_years": 4,
        "pagibig_years": 3,
        "pagibig_status": "Active - 3 years",
        "existing_loans": "Car loan",
        "has_informal_debt": 0,
        "language": "Kapampangan",
        "preferred_location": "San Fernando, Pampanga",
        "submitted_docs": ["Government ID", "PSA Birth Certificate", "ITR", "Pay Slip x3", "Pag-IBIG printout"],
    },
    {
        "id": "HELP-004",
        "name": "Jayson Pineda",
        "role": "Construction Foreman",
        "age": 39,
        "civil_status": "Married",
        "dependents": 2,
        "monthly_income": 24000,
        "employment_status": "Contractual",
        "employment_type_code": "contractual",
        "employment_tenure_years": 2,
        "pagibig_years": 2,
        "pagibig_status": "Active - 2 years",
        "existing_loans": "Motorcycle loan",
        "has_informal_debt": 0,
        "language": "English",
        "preferred_location": "Tarlac",
        "submitted_docs": ["Government ID", "PSA Birth Certificate", "Pay Slip x3"],
    },
    {
        "id": "HELP-005",
        "name": "Teresita Ramos",
        "role": "Sari-sari Store Owner",
        "age": 42,
        "civil_status": "Widowed",
        "dependents": 1,
        "monthly_income": 16000,
        "employment_status": "Self-employed (sari-sari)",
        "employment_type_code": "self_employed",
        "employment_tenure_years": 9,
        "pagibig_years": 1.5,
        "pagibig_status": "Partially active",
        "existing_loans": "None",
        "has_informal_debt": 0,
        "language": "Filipino",
        "preferred_location": "Manila",
        "submitted_docs": ["Government ID", "Business Permit"],
    },
]

CASE_MANAGERS: List[Dict] = [
    {
        "name": "Reynan Tayag",
        "languages": ["Filipino", "English"],
        "areas": ["Bulacan", "Manila", "Pampanga", "Tarlac", "Cebu"],
    },
    {
        "name": "Leo Reyes",
        "languages": ["Kapampangan", "Filipino", "English"],
        "areas": ["Pampanga", "Tarlac"],
    },
    {
        "name": "Ruth Nartatez",
        "languages": ["Bisaya", "Filipino", "English"],
        "areas": ["Cebu", "Davao", "Bohol"],
    },
    {
        "name": "Audrey Yaneza",
        "languages": ["Filipino", "English"],
        "areas": ["Manila", "Bulacan", "Rizal"],
    },
]
CASE_MANAGER_NAMES = [m["name"] for m in CASE_MANAGERS]


def get_api_key() -> str:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.getenv("ANTHROPIC_API_KEY", "")
    return key


API_KEY = get_api_key()
client = anthropic.Anthropic(api_key=API_KEY) if (API_KEY and anthropic is not None) else None
MODEL_CANDIDATES = ["claude-sonnet-4-20250514", "claude-3-5-sonnet-latest"]


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 900, temperature: float = 0.2) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if client is None:
        return None, None, "No API key configured."

    errors: List[str] = []
    for model in MODEL_CANDIDATES:
        for attempt in range(1, 4):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text = response.content[0].text if getattr(response, "content", None) else ""
                return text, model, None
            except Exception as exc:
                errors.append(f"{model} attempt {attempt}: {exc}")
                if attempt < 3:
                    time.sleep(1.5 * (2 ** (attempt - 1)))

    return None, None, " | ".join(errors[-4:])


@st.cache_resource
def load_ml_bundle() -> Dict:
    candidates = [
        Path("homematch_model.pkl"),
        Path(__file__).resolve().parent / "homematch_model.pkl",
        Path(__file__).resolve().parent.parent / "homematch_model.pkl",
    ]

    for model_path in candidates:
        if not model_path.exists():
            continue

        try:
            loaded = joblib.load(model_path)
            if isinstance(loaded, dict) and "model" in loaded:
                return {
                    "model": loaded["model"],
                    "features": loaded.get("features", DEFAULT_FEATURES),
                    "source": str(model_path),
                }

            return {
                "model": loaded,
                "features": DEFAULT_FEATURES,
                "source": str(model_path),
            }
        except Exception as exc:
            return {
                "model": None,
                "features": DEFAULT_FEATURES,
                "source": f"Failed to load {model_path}: {exc}",
            }

    return {
        "model": None,
        "features": DEFAULT_FEATURES,
        "source": "No homematch_model.pkl found. Using deterministic fallback scorer.",
    }


def normalize_docs(profile: Dict) -> List[str]:
    return [d.lower() for d in profile.get("submitted_docs", [])]


def compute_doc_completeness(profile: Dict) -> float:
    text = " | ".join(normalize_docs(profile))
    checks = [
        ["id", "passport"],
        ["psa", "birth certificate"],
        ["marriage", "death", "cenomar"],
        ["itr", "pay slip", "business permit", "employment contract"],
        ["pag-ibig", "hdmf"],
        ["barangay"],
        ["billing", "utility"],
    ]
    hits = sum(1 for group in checks if any(token in text for token in group))
    return hits / 7.0


def profile_to_feature_vector(profile: Dict, feature_names: List[str]) -> np.ndarray:
    feature_map = {
        "monthly_income": float(profile.get("monthly_income", 0)),
        "employment_years": float(profile.get("employment_tenure_years", 0)),
        "dependents": float(profile.get("dependents", 0)),
        "age": float(profile.get("age", 30)),
        "pagibig_years": float(profile.get("pagibig_years", 0)),
        "has_informal_debt": float(profile.get("has_informal_debt", 0)),
        "doc_completeness": float(profile.get("doc_completeness", compute_doc_completeness(profile))),
        "emp_type_score": float(EMP_RISK_MAP.get(profile.get("employment_type_code", "contractual"), 0.4)),
    }
    row = [feature_map.get(f, 0.0) for f in feature_names]
    return np.array([row], dtype=float)


def fallback_score(profile: Dict) -> float:
    income = min(float(profile.get("monthly_income", 0)) / 50000.0, 1.0)
    tenure = min(float(profile.get("employment_tenure_years", 0)) / 10.0, 1.0)
    pagibig = min(float(profile.get("pagibig_years", 0)) / 5.0, 1.0)
    docs = float(profile.get("doc_completeness", compute_doc_completeness(profile)))
    debt_penalty = 0.22 if int(profile.get("has_informal_debt", 0)) == 1 else 0.0
    stability = float(EMP_RISK_MAP.get(profile.get("employment_type_code", "contractual"), 0.4))

    raw = (0.28 * income) + (0.18 * tenure) + (0.22 * pagibig) + (0.20 * docs) + (0.17 * stability) - debt_penalty
    return float(np.clip(raw, 0.05, 0.98))


def predict_readiness(profile: Dict, bundle: Dict) -> float:
    model = bundle.get("model")
    features = bundle.get("features", DEFAULT_FEATURES)

    if model is None:
        return fallback_score(profile)

    try:
        vector = profile_to_feature_vector(profile, features)
        if hasattr(model, "predict_proba"):
            return float(np.clip(model.predict_proba(vector)[0][1], 0.0, 1.0))
        return fallback_score(profile)
    except Exception:
        return fallback_score(profile)


def risk_tier(score: float) -> str:
    if score >= 0.80:
        return "PREFERRED"
    if score >= 0.55:
        return "STANDARD"
    return "NEEDS SUPPORT"


def risk_badge_class(score: float) -> str:
    if score >= 0.80:
        return "risk-preferred"
    if score >= 0.55:
        return "risk-standard"
    return "risk-needs"


def missing_documents(profile: Dict) -> List[str]:
    docs = " | ".join(normalize_docs(profile))
    required_with_matchers = [
        (REQUIRED_DOCS[0], ["id", "passport"]),
        (REQUIRED_DOCS[1], ["psa", "birth certificate"]),
        (REQUIRED_DOCS[2], ["marriage", "death", "cenomar"]),
        (REQUIRED_DOCS[3], ["itr", "pay slip", "business permit", "employment contract"]),
        (REQUIRED_DOCS[4], ["pag-ibig", "hdmf"]),
        (REQUIRED_DOCS[5], ["barangay"]),
        (REQUIRED_DOCS[6], ["billing", "utility"]),
    ]
    return [label for label, matcher in required_with_matchers if not any(tok in docs for tok in matcher)]


def deterministic_letter(profile: Dict, mode: str) -> str:
    missing = missing_documents(profile)
    list_text = "\n".join([f"- {d}" for d in missing]) if missing else "- All primary requirements are complete."
    if "Kapampangan" in mode:
        return (
            f"Mayap a aldo, {profile['name']},\n\n"
            "Iti ing status ning housing application yu.\n"
            "Kailangan mi pa ing kayabe a dokumentu:\n"
            f"{list_text}\n\n"
            "Ngeni mang kumpleto ne, ibie mi ya king case manager para king pinal a beripikasyun.\n\n"
            "Dakal a salamat,\n"
            "Happy Foundation"
        )
    if "Bisaya" in mode:
        return (
            f"Maayong adlaw, {profile['name']},\n\n"
            "Mao ni ang kasamtangang status sa imong housing application.\n"
            "Palihug isumite ang kulang nga dokumento:\n"
            f"{list_text}\n\n"
            "Kung kompleto na, among ipadala sa case manager para sa final nga validation.\n\n"
            "Daghang salamat,\n"
            "Happy Foundation"
        )
    if "Filipino" in mode:
        return (
            f"Mahal na {profile['name']},\n\n"
            "Narito ang kasalukuyang status ng inyong housing application.\n"
            "Kailangan pa po namin ang mga sumusunod:\n"
            f"{list_text}\n\n"
            "Kapag kumpleto na ang dokumento, ipapasa namin ito para sa final validation ng case manager.\n\n"
            "Lubos na gumagalang,\n"
            "Happy Foundation"
        )

    return (
        f"Dear {profile['name']},\n\n"
        "Here is your current housing application status.\n"
        "Please submit the following pending requirements:\n"
        f"{list_text}\n\n"
        "Once complete, we will route your file for final case-manager validation.\n\n"
        "Regards,\n"
        "Happy Foundation"
    )


def render_topbar(title: str, subtitle: str) -> str:
    st.markdown(f"<div class='top-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='top-sub'>{subtitle}</div>", unsafe_allow_html=True)


bundle = load_ml_bundle()

profiles = []
for raw in APPLICANTS:
    p = raw.copy()
    p["doc_completeness"] = compute_doc_completeness(p)
    p["score"] = predict_readiness(p, bundle)
    p["tier"] = risk_tier(p["score"])
    profiles.append(p)

profile_df = pd.DataFrame(profiles)

with st.sidebar:
    st.markdown("<div class='sidebar-logo'>AbotBahay</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>HAPPY FOUNDATION PARTNER</div>", unsafe_allow_html=True)

    st.markdown("**OPERATIONS**")
    page = st.radio("", ["Applicant Pipeline", "Document Drafter"], label_visibility="collapsed")

    st.markdown("<div class='system-card'><div style='font-size:0.85rem; color:#f8bb47; font-weight:700;'>SYSTEM ONLINE</div><div style='font-size:0.9rem; margin-top:6px;'>Localized Intelligence</div></div>", unsafe_allow_html=True)
    st.caption(bundle["source"])

right1, right2, right3 = st.columns([6.5, 1.5, 1.8])
with right2:
    language_ui = st.selectbox("Language", ["English", "Filipino", "Kapampangan", "Bisaya"], label_visibility="collapsed")
with right3:
    st.markdown("<div style='text-align:right; font-size:0.75rem; color:#8aa0bf; font-weight:700;'>CASE MANAGER</div>", unsafe_allow_html=True)
    selected_case_manager = st.selectbox(
        "Case Manager",
        CASE_MANAGER_NAMES,
        index=0,
        key="case_manager_selector",
        label_visibility="collapsed",
    )

if page == "Applicant Pipeline":
    render_topbar("Applicant Readiness Overview", "Coordinated by Happy Foundation | Housing Program Management")

    success_prob = profile_df["score"].mean() * 100
    match_success = profile_df["score"].median()
    active_families = len(profile_df)
    completion_rate = (profile_df["doc_completeness"] >= 0.5).mean() * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>Success Prob.</div><div class='kpi-value'>{success_prob:.1f}%</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>Match Success</div><div class='kpi-value' style='color:#c96a00'>{match_success:.2f}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>Active Families</div><div class='kpi-value'>{active_families}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi'><div class='kpi-label'>Completion Rate</div><div class='kpi-value' style='color:#0f9f6e'>{completion_rate:.0f}%</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='panel' style='margin-top:14px;'><div class='panel-head'><div style='font-size:1.9rem; font-weight:800;'>Applicant Readiness Matrix</div><div style='color:#60738f'>Intelligent matching for housing eligibility</div></div><div class='panel-body'>", unsafe_allow_html=True)

    header = st.columns([2.4, 1.4, 1.8, 1.1, 1.1])
    header[0].markdown("**Applicant Profile**")
    header[1].markdown("**Preferred Language**")
    header[2].markdown("**Match Readiness**")
    header[3].markdown("**Risk Tier**")
    header[4].markdown("**Operations**")

    for _, row in profile_df.sort_values("score", ascending=False).iterrows():
        cols = st.columns([2.4, 1.4, 1.8, 1.1, 1.1])
        cols[0].markdown(f"**{row['name']}**  \n<span style='color:#71839f'>{row['role']}</span>", unsafe_allow_html=True)
        cols[1].markdown(f"**{row['language']}**  \n<span style='color:#71839f'>{row['preferred_location']}</span>", unsafe_allow_html=True)
        with cols[2]:
            st.markdown(f"**{row['score']*100:.0f}%**")
            st.progress(float(row["score"]))
        cols[3].markdown(
            f"<span class='risk-badge {risk_badge_class(row['score'])}'>{row['tier']}</span>",
            unsafe_allow_html=True,
        )
        cols[4].button("View Profile", key=f"view_{row['id']}")

    st.markdown("</div></div>", unsafe_allow_html=True)

else:
    render_topbar("Document Drafter Tool", "Coordinated by Happy Foundation | Housing Program Management")

    selected_name = st.selectbox("Applicant", profile_df["name"].tolist(), index=1, key="drafter_applicant")
    selected = next(p for p in profiles if p["name"] == selected_name)

    left_col, right_col = st.columns([1.1, 1.9])

    with left_col:
        initials = "".join([part[0] for part in selected["name"].split()[:2]]).upper()
        st.markdown(
            f"""
<div class='applicant-card'>
  <div style='font-size:0.9rem; color:#8aa0bf; font-weight:700;'>APPLICANT DATA</div>
  <div class='avatar'>{initials}</div>
  <div style='text-align:center; font-size:2rem; font-weight:800;'>{selected['name']}</div>
  <div style='text-align:center; color:#7d90ab; font-weight:700; letter-spacing:0.03em;'>{selected['role']}</div>
  <hr style='margin:16px 0; border:0; border-top:1px solid #e4e9f2;'>
  <div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span style='color:#627894;'>MONTHLY INCOME</span><b>PHP {selected['monthly_income']:,.0f}</b></div>
  <div style='display:flex; justify-content:space-between; margin-bottom:8px;'><span style='color:#627894;'>LOCATION</span><b>{selected['preferred_location']}</b></div>
  <div style='display:flex; justify-content:space-between;'><span style='color:#627894;'>MATCH SCORE</span><b style='color:#c96a00'>{selected['score']*100:.0f}%</b></div>
</div>
""",
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        hdr1, hdr2, hdr3 = st.columns([2.2, 1.2, 1.0])
        with hdr1:
            st.markdown("### Application Drafter")
        with hdr2:
            draft_choices = ["Draft in English", "Draft in Filipino", "Draft in Kapampangan", "Draft in Bisaya"]
            preferred = f"Draft in {selected['language']}" if f"Draft in {selected['language']}" in draft_choices else "Draft in English"
            default_idx = draft_choices.index(preferred)
            draft_mode = st.selectbox("", draft_choices, index=default_idx, label_visibility="collapsed")
        with hdr3:
            draft_now = st.button("Draft Letter", type="primary", use_container_width=True)

        if "draft_output" not in st.session_state:
            st.session_state.draft_output = "Document content will appear here..."

        if draft_now:
            system_prompt = (
                "You are HomeMatch Document Drafter. Create professional, clear follow-up letters. "
                "Do not guarantee housing approval. "
                "Write in the requested language or dialect. If dialect confidence is low, keep terms simple and note for staff review."
            )
            prompt = (
                f"Applicant: {selected['name']} ({selected['role']}) | Location: {selected['preferred_location']} | "
                f"Readiness: {selected['score']*100:.0f}% | Missing docs: {', '.join(missing_documents(selected))} | "
                f"Draft mode: {draft_mode}. Assigned case manager: {selected_case_manager}."
            )
            llm_text, llm_model, llm_error = call_llm(system_prompt, prompt, max_tokens=700, temperature=0.25)
            if llm_error:
                st.session_state.draft_output = deterministic_letter(selected, draft_mode) + f"\n\n[Fallback mode: {llm_error}]"
            else:
                st.session_state.draft_output = llm_text + f"\n\n[Model: {llm_model}]"

        st.text_area("", value=st.session_state.draft_output, height=540, label_visibility="collapsed")

        ft1, ft2, ft3 = st.columns([3.1, 1.0, 1.2])
        with ft1:
            st.markdown("<span style='color:#8aa0bf; font-weight:700;'>POWERED BY ABOT BAHAY INTELLIGENCE</span>", unsafe_allow_html=True)
        with ft2:
            st.button("Copy Text", use_container_width=True)
        with ft3:
            st.button("Approve & Send", type="primary", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)
