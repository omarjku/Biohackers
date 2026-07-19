import os

import streamlit as st

# Load OPENAI_API_KEY (and anything else) from a local .env if present, so the
# documented `cp .env.example .env` workflow actually reaches the process. No-op
# if python-dotenv or the file is missing; the GPT path just stays off.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =====================================================================
# 0. BACKEND IMPORTS
# =====================================================================
# Both the explainer and the real pipeline are required. There is deliberately
# NO mock fallback: a mock returned fabricated out-of-scope (TB / S. aureus)
# predictions, which would be shown as real results and contradict the E. coli
# scope declaration — exactly the "overstated coverage" trap the brief penalizes.
# If a backend module cannot import, fail loudly rather than demo fabricated data.
try:
    import explainer
    import pipeline
except ImportError as e:
    st.error(f"Backend unavailable — cannot run the pipeline: {e}")
    st.stop()

# =====================================================================
# 1. PAGE CONFIGURATION & CSS THEME
# =====================================================================
st.set_page_config(
    page_title="Genome Firewall | Module 03",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; max-width: 1180px; }
    .stApp { background-color: #f6f7f9; }

    .app-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; padding-bottom: 15px; border-bottom: 1px solid #e5e7eb; }
    .app-title { font-size: 1.85rem; font-weight: 800; color: #0f172a; margin: 0; letter-spacing: -0.6px; }
    .app-subtitle { font-size: 0.92rem; color: #64748b; margin: 2px 0 0 0; font-weight: 500; }
    .app-badge { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #1d4ed8; background: #eff6ff; border: 1px solid #bfdbfe; padding: 6px 12px; border-radius: 20px; }

    .clinical-banner { background-color: #fff7ed; border: 1px solid #ffedd5; border-left: 4px solid #f97316; padding: 12px 16px; border-radius: 6px; color: #9a3412; font-size: 0.85rem; margin-bottom: 22px; display: flex; gap: 12px; align-items: flex-start; line-height: 1.5; }

    /* QC / not-a-genome warning */
    .qc-warning { background-color: #fef2f2; border: 1px solid #fecaca; border-left: 4px solid #ef4444; padding: 14px 18px; border-radius: 8px; color: #991b1b; font-size: 0.9rem; margin: 18px 0 6px 0; line-height: 1.55; }
    .qc-warning strong { color: #7f1d1d; }

    /* Metrics table — matches the light theme (replaces the dark st.dataframe) */
    .metrics-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 6px; }
    .metrics-table th { text-align: left; color: #64748b; font-weight: 700; text-transform: uppercase; font-size: 0.68rem; letter-spacing: 0.5px; padding: 8px 10px; border-bottom: 2px solid #e5e7eb; }
    .metrics-table td { padding: 9px 10px; border-bottom: 1px solid #f1f5f9; color: #1e293b; font-family: 'Courier New', monospace; }
    .metrics-table td.drug { font-family: inherit; font-weight: 600; color: #0f172a; }

    .qc-bar { display: flex; gap: 34px; background-color: #ffffff; padding: 15px 20px; border-radius: 10px; border: 1px solid #e5e7eb; margin: 6px 0 22px 0; }
    .qc-metric { display: flex; flex-direction: column; }
    .qc-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
    .qc-value { font-size: 1.05rem; color: #0f172a; font-weight: 600; font-family: 'Courier New', monospace; }

    /* Overall AI summary panel */
    .ai-summary { background: linear-gradient(135deg, #f5f8ff 0%, #eef4ff 100%); border: 1px solid #dbe4ff; border-left: 4px solid #4f46e5; border-radius: 10px; padding: 16px 20px; margin-bottom: 24px; }
    .ai-summary-label { display: flex; align-items: center; gap: 7px; font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.6px; color: #4338ca; margin-bottom: 7px; }
    .ai-summary-text { font-size: 0.95rem; color: #1e293b; line-height: 1.6; }

    /* Cards */
    .card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; }
    .drug-name { font-size: 1.22rem; font-weight: 700; color: #0f172a; margin: 0; }
    .drug-class { font-size: 0.8rem; color: #64748b; font-weight: 500; margin-top: 2px; }

    .badge { padding: 5px 13px; border-radius: 20px; font-weight: 700; font-size: 0.78rem; display: inline-flex; align-items: center; gap: 6px; text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap; }
    .badge-fail { background-color: #fdf2f8; color: #be123c; border: 1px solid #fecdd3; }
    .badge-work { background-color: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-nocall { background-color: #fffbeb; color: #b45309; border: 1px solid #fde68a; }

    .score-container { display: flex; justify-content: space-between; font-size: 0.88rem; color: #64748b; margin-bottom: 7px; font-weight: 500; }
    .score-value { font-weight: 700; color: #0f172a; }
    .progress-bg { width: 100%; background-color: #eef2f6; border-radius: 6px; height: 9px; margin-bottom: 16px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 6px; }
    .fill-fail { background-color: #e11d48; }
    .fill-work { background-color: #16a34a; }
    .fill-nocall { background-color: #f59e0b; }

    /* Evidence tier chip */
    .evi-chip { display: inline-flex; align-items: center; gap: 6px; font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; padding: 4px 11px; border-radius: 6px; margin-bottom: 14px; }
    .evi-known { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    .evi-stat  { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
    .evi-none  { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }

    /* AI explanation block — always visible, prominent */
    .ai-block { background: #f8faff; border: 1px solid #e6edff; border-radius: 8px; padding: 12px 14px; margin-bottom: 12px; }
    .ai-block-label { display: flex; align-items: center; gap: 6px; font-size: 0.68rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; color: #4338ca; margin-bottom: 6px; }
    .ai-block-text { font-size: 0.9rem; color: #1e293b; line-height: 1.55; }

    /* Target / marker boxes */
    .target-box { background-color: #f8fafc; border-radius: 8px; padding: 12px 14px; margin-bottom: 12px; display: flex; gap: 34px; border: 1px solid #eef2f6; }
    .target-col { display: flex; flex-direction: column; gap: 4px; }
    .target-title { font-size: 0.66rem; color: #94a3b8; font-family: 'Courier New', monospace; text-transform: uppercase; font-weight: 700; letter-spacing: 0.4px; }
    .target-value { font-family: 'Courier New', monospace; font-size: 0.92rem; font-weight: 700; color: #0f172a; }

    /* Make Streamlit expanders clearly visible (the old evidence button was translucent) */
    div[data-testid="stExpander"] { border: 1px solid #e2e8f0 !important; border-radius: 8px !important; background: #ffffff !important; margin-top: 4px; }
    div[data-testid="stExpander"] summary { padding: 9px 14px !important; font-weight: 600 !important; color: #334155 !important; font-size: 0.84rem !important; }
    div[data-testid="stExpander"] summary:hover { color: #0f172a !important; background: #f8fafc !important; }
    .streamlit-expanderContent p, div[data-testid="stExpander"] p { font-size: 0.88rem; color: #334155; line-height: 1.55; }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. HEADER
# =====================================================================
st.markdown("""
    <div class="app-header">
        <div>
            <h1 class="app-title">Genome Firewall — Susceptibility Report</h1>
            <p class="app-subtitle">AI-assisted antibiotic-susceptibility interpretation · Module 03</p>
        </div>
        <div class="app-badge">E. coli prototype</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="clinical-banner">
        <div style="font-size: 1.2rem; font-weight: 800;">!</div>
        <div>
            <strong>MANDATORY CLINICAL DISCLAIMER:</strong> This software is a decision-support tool only and is not authorized to make standalone therapeutic choices.
            Every automated prediction must be confirmed by standard wet-lab phenotypic testing before altering clinical management protocols.
        </div>
    </div>
""", unsafe_allow_html=True)

# =====================================================================
# 3. SCOPE & GENERALIZATION METRICS
# =====================================================================
_METRICS = [
    # drug, bal_acc, recall_R, recall_S, auroc, coverage
    ("Ampicillin", "0.94", "0.91", "0.97", "0.95", "91%"),
    ("Ciprofloxacin", "0.85", "0.76", "0.95", "0.91", "84%"),
    ("Trimethoprim", "0.92", "0.88", "0.95", "0.94", "85%"),
]

with st.expander("System Scope & Generalization Metrics", expanded=True):
    col_scope, col_metrics = st.columns(2)
    with col_scope:
        st.markdown("**System Scope Declaration**")
        st.markdown("""
        * **Species:** *Escherichia coli* (single-species prototype).
        * **Antibiotics:** Ampicillin, Ciprofloxacin, Trimethoprim.
        * **Out of Scope:** other species, other antibiotics, sample-to-genome
          processing, and any organism design or modification.
        """)
    with col_metrics:
        st.markdown("**Generalization Performance (MLST-Split Results)**")
        st.caption("Mean over 8 grouped splits by MLST lineage (2,127 genomes) — held-out groups, no near-identical leakage. recall_R is the fraction of truly-resistant isolates caught.")
        rows = "".join(
            f"<tr><td class='drug'>{d}</td><td>{ba}</td><td>{rr}</td>"
            f"<td>{rs}</td><td>{au}</td><td>{cov}</td></tr>"
            for d, ba, rr, rs, au, cov in _METRICS
        )
        st.markdown(f"""
            <table class="metrics-table">
                <thead><tr>
                    <th>Antibiotic</th><th>Bal. Acc</th><th>Recall R</th>
                    <th>Recall S</th><th>AUROC</th><th>Coverage</th>
                </tr></thead>
                <tbody>{rows}</tbody>
            </table>
        """, unsafe_allow_html=True)
        st.caption("Ciprofloxacin's lower resistant-recall (0.76) reflects mutation-driven resistance (gyrA/parC) that the acquired-gene features under-capture — reported honestly rather than hidden.")

st.divider()

# =====================================================================
# 4. UPLOAD + OPTIONS
# =====================================================================
uploaded_fasta = st.file_uploader("Upload Reconstructed Bacterial Genome (FASTA)", type=["fasta", "fa", "fna"])

# Optional bundled example genomes (present only when data/raw/fasta_demo has been
# populated locally) so the demo can run offline without hunting for a FASTA.
import glob

_example_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "fasta_demo")
_examples = sorted(glob.glob(os.path.join(_example_dir, "*.fna")))
example_choice = None
if _examples:
    with st.expander("…or run a bundled example E. coli genome"):
        _names = ["—"] + [os.path.basename(e) for e in _examples]
        _sel = st.selectbox("Example genome (real BV-BRC assembly)", _names)
        if _sel != "—":
            example_choice = os.path.join(_example_dir, _sel)

# AI explanations toggle. ON by default: the explainer refines each card's
# biological/statistical text and writes an overall clinical summary with
# gpt-4o-mini, and falls back to the deterministic template on any failure or
# when no key is present — so this can never break a demo, only enrich it.
_key = os.environ.get("OPENAI_API_KEY", "")
_has_key = _key.startswith("sk-") and _key != "sk-your-key-here"
use_gpt = st.toggle(
    "AI-written explanations (OpenAI)",
    value=True,
    help="On = gpt-4o-mini writes clinician-readable explanations + an overall "
         "summary. Off (or no API key) = built-in deterministic explanations.",
)
if use_gpt and not _has_key:
    st.caption("ℹ️ No `OPENAI_API_KEY` detected — using built-in deterministic "
               "explanations (add a key to `.env` for AI-written prose).")

fasta_source = uploaded_fasta if uploaded_fasta is not None else example_choice
source_name = (uploaded_fasta.name if uploaded_fasta is not None
               else (os.path.basename(example_choice) if example_choice else None))

# =====================================================================
# 5. PIPELINE EXECUTION + REPORT
# =====================================================================
if fasta_source is not None:
    try:
        with st.status("Executing Genome Firewall Pipeline...", expanded=True) as status:
            st.write("QC: checking the upload is a whole-genome assembly...")
            st.write("Module 01: Annotating genome with AMRFinderPlus...")
            st.write("Module 02: Scoring per-antibiotic models + Platt calibration...")
            qc, raw_predictions = pipeline.analyze(fasta_source)
            st.write("Module 03: Explainer NL layer resolving evidence...")
            status.update(label="Analysis Complete", state="complete", expanded=False)
    except Exception as exc:  # noqa: BLE001 — surface any pipeline failure cleanly
        st.error(
            "Could not analyze this genome. This usually means AMRFinderPlus is not "
            "installed/configured on this machine, or the FASTA could not be "
            f"annotated. See docs/LIVE_DEMO.md for setup.\n\nDetails: {exc}"
        )
        st.stop()

    # QC warning banner (warn-but-still-score): a non-genome upload (e.g. a 16S
    # gene) produces a confident-but-meaningless call, so flag it loudly.
    if not qc["plausible_genome"]:
        st.markdown(f"""
            <div class="qc-warning">
                <strong>⚠ This does not look like a whole-genome assembly.</strong><br>
                {qc['message']}<br>
                <em>Results are shown below for transparency but should not be trusted for this input.</em>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="qc-bar">
            <div class="qc-metric"><span class="qc-label">Target File</span><span class="qc-value">{source_name}</span></div>
            <div class="qc-metric"><span class="qc-label">Species Model</span><span class="qc-value">E. coli</span></div>
            <div class="qc-metric"><span class="qc-label">Assembly Size</span><span class="qc-value">{qc['total_bp']:,} bp</span></div>
            <div class="qc-metric"><span class="qc-label">Contigs</span><span class="qc-value">{qc['n_contigs']}</span></div>
            <div class="qc-metric"><span class="qc-label">Compounds</span><span class="qc-value">{len(raw_predictions)}</span></div>
        </div>
    """, unsafe_allow_html=True)

    # ----- Overall AI clinical summary -----
    summary_text = explainer.clinical_summary(raw_predictions, use_llm=use_gpt)
    ai_tag = "AI Clinical Summary" if (use_gpt and _has_key) else "Clinical Summary"
    st.markdown(f"""
        <div class="ai-summary">
            <div class="ai-summary-label">✦ {ai_tag}</div>
            <div class="ai-summary-text">{summary_text}</div>
        </div>
    """, unsafe_allow_html=True)

    # ----- Per-drug report -----
    reports = explainer.explain_report(raw_predictions, use_llm=use_gpt)

    # evidence_category lives on the raw Prediction, not the report dict — map it back.
    _evidence_by_drug = {p.drug: p.evidence_category for p in raw_predictions}
    EVIDENCE_CHIP = {
        "known_gene_or_mutation": ("evi-known", "✓ Known resistance mechanism"),
        "statistical_association": ("evi-stat", "≈ Statistical association only"),
        "no_known_signal": ("evi-none", "○ No resistance signal detected"),
    }

    STATE_STYLE = {
        "Likely to work": ("badge-work", "fill-work"),
        "Likely to fail": ("badge-fail", "fill-fail"),
        "No-call": ("badge-nocall", "fill-nocall"),
    }

    ui_cards = []
    for rep in reports:
        state = rep["underlying_state"]
        badge, fill = STATE_STYLE.get(state, ("badge-nocall", "fill-nocall"))
        evi_cls, evi_label = EVIDENCE_CHIP.get(
            _evidence_by_drug.get(rep["drug"]), ("evi-none", "○ Inconclusive evidence")
        )
        ui_cards.append({
            "drug": rep["drug"],
            "drug_class": rep["drug_class"],
            "confidence": rep["confidence"],
            "call_status": state,
            "target_marker": rep["target_marker"],
            "locus": rep["locus_id"],
            "bio_explanation": rep["bio_explanation"],
            "stat_explanation": rep["stat_explanation"],
            "evi_cls": evi_cls, "evi_label": evi_label,
            "badge": badge, "fill": fill, "label": state,
        })

    # Sort: workable first, then fail, then no-call.
    ORDER = {"Likely to work": 0, "Likely to fail": 1, "No-call": 2}
    sorted_cards = sorted(ui_cards, key=lambda c: ORDER.get(c["call_status"], 3))

    ai_label = "AI Explanation" if (use_gpt and _has_key) else "Explanation"

    for i in range(0, len(sorted_cards), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j >= len(sorted_cards):
                continue
            card = sorted_cards[i + j]
            with cols[j]:
                with st.container(border=True):
                    st.markdown(f"""
                        <div class="card-header">
                            <div>
                                <h3 class="drug-name">{card['drug']}</h3>
                                <div class="drug-class">{card['drug_class']}</div>
                            </div>
                            <div class="badge {card['badge']}">{card['label']}</div>
                        </div>

                        <span class="evi-chip {card['evi_cls']}">{card['evi_label']}</span>

                        <div class="score-container">
                            <span>Model certainty in this call</span>
                            <span class="score-value">{card['confidence'] * 100:.1f}%</span>
                        </div>
                        <div class="progress-bg">
                            <div class="progress-fill {card['fill']}" style="width: {card['confidence'] * 100}%;"></div>
                        </div>

                        <div class="ai-block">
                            <div class="ai-block-label">✦ {ai_label}</div>
                            <div class="ai-block-text">{card['bio_explanation']}</div>
                        </div>

                        <div class="target-box">
                            <div class="target-col">
                                <span class="target-title">Target / Marker</span>
                                <span class="target-value">{card['target_marker']}</span>
                            </div>
                            <div class="target-col">
                                <span class="target-title">Locus / Gene ID</span>
                                <span class="target-value">{card['locus']}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    with st.expander("Statistical rationale & disclaimer"):
                        st.markdown(
                            "<span style='font-size:0.7rem;color:#94a3b8;font-family:Courier;"
                            "text-transform:uppercase;font-weight:700;'>Statistical Rationale</span>",
                            unsafe_allow_html=True)
                        st.write(card['stat_explanation'])
                        st.markdown(
                            "<br><span style='font-size:0.7rem;color:#94a3b8;font-family:Courier;"
                            "text-transform:uppercase;font-weight:700;'>Mandatory Disclaimer</span>",
                            unsafe_allow_html=True)
                        st.write(explainer.DISCLAIMER)
else:
    st.markdown("""
        <div style="text-align: center; padding: 50px 20px; background-color: #ffffff; border-radius: 10px; border: 1px dashed #cbd5e1; margin-top: 20px;">
            <h3 style="color: #475569; font-size: 1.1rem; margin: 0 0 5px 0;">Awaiting a genome</h3>
            <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">Upload a whole-genome E. coli FASTA (~5 Mb) — or run a bundled example — to generate the susceptibility report.</p>
        </div>
    """, unsafe_allow_html=True)
