import streamlit as st
import pandas as pd
import time

# =====================================================================
# 0. GRACEFUL IMPORTS & MOCK FALLBACKS
# =====================================================================
# 1. Load Hazem's Explainer and Schemas (Imported as direct siblings)
try:
    from schemas import Prediction, ExplanationResult, SupportingFeature
    import explainer

    EXPLAINER_CONNECTED = True
except ImportError as e:
    st.error(f"Import Error: {e}")
    st.stop()

# 2. Mock the missing Pipeline
try:
    # If your team eventually creates a pipeline.py, it will connect automatically
    import pipeline

    PIPELINE_CONNECTED = True
except ImportError:
    PIPELINE_CONNECTED = False


    # Generate fake backend data using the real schemas so your UI works today
    class MockPipeline:
        @staticmethod
        def run(fasta_file):
            return [
                Prediction(
                    sample_id="SEQ-001", species="M. tuberculosis", drug="Isoniazid",
                    call="likely_to_fail", confidence=0.974, evidence_category="known_gene_or_mutation",
                    target_gate_status="present",
                    supporting_features=[SupportingFeature(gene="katG", mutation="S315T")],
                    no_call_reason=None
                ),
                Prediction(
                    sample_id="SEQ-001", species="M. tuberculosis", drug="Ciprofloxacin",
                    call="likely_to_work", confidence=0.942, evidence_category="no_known_signal",
                    target_gate_status="present", supporting_features=[],
                    no_call_reason=None
                ),
                Prediction(
                    sample_id="SEQ-001", species="M. tuberculosis", drug="Ethambutol",
                    call="no_call", confidence=0.421, evidence_category="statistical_association",
                    target_gate_status="present", supporting_features=[],
                    no_call_reason="Ambiguous structural resolution"
                ),
                Prediction(
                    sample_id="SEQ-001", species="S. aureus", drug="Methicillin",
                    call="not_applicable", confidence=1.0, evidence_category="no_known_signal",
                    target_gate_status="absent", supporting_features=[],
                    no_call_reason=None
                )
            ]


    pipeline = MockPipeline()

# =====================================================================
# 1. PAGE CONFIGURATION & CSS THEME
# =====================================================================
st.set_page_config(
    page_title="Genome Firewall | Module 03",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container { padding-top: 2rem !important; padding-bottom: 2rem !important; max-width: 1200px; }
    .stApp { background-color: #f8f9fa; }

    .app-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #e5e7eb; }
    .app-title { font-size: 1.8rem; font-weight: 800; color: #111827; margin: 0; letter-spacing: -0.5px; }
    .app-subtitle { font-size: 0.95rem; color: #6b7280; margin: 0; font-weight: 500; }

    .clinical-banner { background-color: #fff7ed; border-left: 4px solid #f97316; padding: 12px 16px; border-radius: 6px; color: #9a3412; font-size: 0.85rem; margin-bottom: 25px; display: flex; gap: 12px; align-items: flex-start; line-height: 1.5; border: 1px solid #ffedd5;}

    .qc-bar { display: flex; gap: 30px; background-color: #ffffff; padding: 15px 20px; border-radius: 8px; border: 1px solid #e5e7eb; margin-bottom: 25px; }
    .qc-metric { display: flex; flex-direction: column; }
    .qc-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
    .qc-value { font-size: 1.1rem; color: #111827; font-weight: 600; font-family: 'Courier New', Courier, monospace; }

    .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
    .drug-name { font-size: 1.25rem; font-weight: 700; color: #111827; margin: 0; }

    .badge { padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; display: inline-flex; align-items: center; gap: 6px; text-transform: uppercase; letter-spacing: 0.5px;}
    .badge-fail { background-color: #fdf2f8; color: #be123c; border: 1px solid #fecdd3; }
    .badge-work { background-color: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-nocall { background-color: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
    .badge-na { background-color: #f3f4f6; color: #4b5563; border: 1px solid #d1d5db; }

    .score-container { display: flex; justify-content: space-between; font-size: 0.9rem; color: #6b7280; margin-bottom: 8px; font-weight: 500; }
    .score-value { font-weight: 700; color: #111827; }

    .progress-bg { width: 100%; background-color: #f3f4f6; border-radius: 6px; height: 8px; margin-bottom: 20px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 6px; transition: width 0.3s ease; }
    .fill-fail { background-color: #e11d48; }
    .fill-work { background-color: #16a34a; }
    .fill-nocall { background-color: #f59e0b; }
    .fill-na { background-color: #9ca3af; }

    .reasoning-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; margin-top: 5px; border-bottom: 1px dashed #e5e7eb; padding-bottom: 10px;}
    .reasoning-label { font-size: 0.75rem; color: #9ca3af; font-family: 'Courier New', Courier, monospace; letter-spacing: 0.5px; text-transform: uppercase; font-weight: 600; }
    .reasoning-value { font-size: 0.85rem; font-weight: 600; color: #374151;}

    .target-box { background-color: #f9fafb; border-radius: 8px; padding: 15px; margin-bottom: 15px; display: flex; gap: 40px; border: 1px solid #f3f4f6; }
    .target-col { display: flex; flex-direction: column; gap: 5px; }
    .target-title { font-size: 0.7rem; color: #9ca3af; font-family: 'Courier New', Courier, monospace; text-transform: uppercase; font-weight: 600; }
    .target-value { font-family: 'Courier New', Courier, monospace; font-size: 0.95rem; font-weight: 700; color: #111827; }

    .streamlit-expanderContent p { font-size: 0.9rem; color: #374151; line-height: 1.5; }

    /* Calibration Controls Customization */
    div[data-testid="stSlider"] label { font-family: 'Courier New', Courier, monospace; font-size: 0.8rem !important; color: #4b5563 !important; text-transform: uppercase; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. HEADER & CALIBRATION
# =====================================================================
st.markdown("""
    <div class="app-header">
        <div>
            <h1 class="app-title">Susceptibility Interpretations</h1>
            <p class="app-subtitle">Dynamic thresholding interface for Genome Firewall Module 03</p>
        </div>
    </div>
""", unsafe_allow_html=True)

col_slider1, col_slider2 = st.columns(2)
with col_slider1:
    decision_threshold = st.slider("Decision Threshold (%)", min_value=50.0, max_value=99.9, value=85.0, step=0.1,
                                   help="Confidence scores below this limit trigger a safety No-call.")
with col_slider2:
    st.slider("No-Call Margin (±%)", min_value=0.0, max_value=20.0, value=10.0, step=0.5, disabled=True,
              help="Margin parameter reserved for backend calibration (Module 02).")

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
# 3. RESPONSIBILITY METRICS & SCOPE
# =====================================================================
with st.expander("System Scope & Generalization Metrics", expanded=False):
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
        st.caption("Offline evaluation over 8 grouped splits by MLST lineage (2,127 genomes) — held-out groups, no near-identical leakage.")
        metrics_data = {
            "Antibiotic": ["Ampicillin", "Ciprofloxacin", "Trimethoprim"],
            "Balanced Acc": ["0.93", "0.85", "0.92"],
            "AUROC": ["0.95", "0.91", "0.94"],
            "Coverage": ["91%", "84%", "85%"],
        }
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)
st.divider()

# =====================================================================
# 4. SEQUENCE PARSING & PIPELINE EXECUTION
# =====================================================================
uploaded_fasta = st.file_uploader("Upload Reconstructed Bacterial Genome (FASTA)", type=["fasta", "fa"])

# Optional bundled example genomes (present only when data/raw/fasta_demo has been
# populated locally) so the demo can run offline without hunting for a FASTA.
import glob
import os

_example_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "fasta_demo")
_examples = sorted(glob.glob(os.path.join(_example_dir, "*.fna")))
example_choice = None
if _examples:
    with st.expander("…or run a bundled example E. coli genome"):
        _names = ["—"] + [os.path.basename(e) for e in _examples]
        _sel = st.selectbox("Example genome (real BV-BRC assembly)", _names)
        if _sel != "—":
            example_choice = os.path.join(_example_dir, _sel)

fasta_source = uploaded_fasta if uploaded_fasta is not None else example_choice
source_name = (uploaded_fasta.name if uploaded_fasta is not None
               else (os.path.basename(example_choice) if example_choice else None))

if fasta_source is not None:
    with st.status("Executing Genome Firewall Pipeline...", expanded=True) as status:
        st.write("Module 01: Annotating genome with AMRFinderPlus...")
        st.write("Module 02: Scoring per-antibiotic models + Platt calibration...")
        raw_predictions = pipeline.run(fasta_source)
        st.write("Module 03: Explainer NL layer resolving evidence...")
        status.update(label="Analysis Complete", state="complete", expanded=False)

    st.markdown(f"""
        <div class="qc-bar">
            <div class="qc-metric"><span class="qc-label">Target File</span><span class="qc-value">{source_name}</span></div>
            <div class="qc-metric"><span class="qc-label">Integration Status</span><span class="qc-value" style="color: {'#15803d' if PIPELINE_CONNECTED else '#b45309'};">{'CONNECTED' if PIPELINE_CONNECTED else 'MOCK ISOLATION'}</span></div>
            <div class="qc-metric"><span class="qc-label">Compounds Evaluated</span><span class="qc-value">{len(raw_predictions)}</span></div>
        </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # NEW-JSON REPORT MAPPING + DECISION-THRESHOLD INTERCEPT
    # -------------------------------------------------------------
    # explain_report() returns the frontend contract: underlying_state,
    # confidence-in-the-call, target_marker, locus_id, drug_class, and the
    # separate biological vs statistical explanations.
    reports = explainer.explain_report(raw_predictions, use_llm=False)

    STATE_STYLE = {
        "Likely to work": ("badge-work", "fill-work"),
        "Likely to fail": ("badge-fail", "fill-fail"),
        "No-call": ("badge-nocall", "fill-nocall"),
    }

    ui_cards = []
    for rep in reports:
        state = rep["underlying_state"]
        conf_pct = rep["confidence"] * 100
        stat_text = rep["stat_explanation"]

        # Decision-threshold slider: abstain when certainty in a definite call
        # falls below the user's chosen threshold.
        if state in ("Likely to work", "Likely to fail") and conf_pct < decision_threshold:
            state = "No-call"
            stat_text = (f"Model certainty {conf_pct:.1f}% is below the decision "
                         f"threshold ({decision_threshold:.0f}%); withheld as a No-call.")

        badge, fill = STATE_STYLE.get(state, ("badge-nocall", "fill-nocall"))
        ui_cards.append({
            "drug": rep["drug"],
            "drug_class": rep["drug_class"],
            "confidence": rep["confidence"],
            "call_status": state,
            "target_marker": rep["target_marker"],
            "locus": rep["locus_id"],
            "bio_explanation": rep["bio_explanation"],
            "stat_explanation": stat_text,
            "disclaimer": explainer.DISCLAIMER,
            "badge": badge, "fill": fill, "label": state,
        })

    # Sort: workable first, then fail, then no-call.
    ORDER = {"Likely to work": 0, "Likely to fail": 1, "No-call": 2}
    sorted_cards = sorted(ui_cards, key=lambda c: ORDER.get(c["call_status"], 3))

    # -------------------------------------------------------------
    # RENDER GRID
    # -------------------------------------------------------------
    for i in range(0, len(sorted_cards), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(sorted_cards):
                card = sorted_cards[i + j]

                with cols[j]:
                    with st.container(border=True):
                        st.markdown(f"""
                            <div class="card-header">
                                <div>
                                    <h3 class="drug-name">{card['drug']}</h3>
                                    <div style="font-size:0.8rem;color:#6b7280;font-weight:500;">{card['drug_class']}</div>
                                </div>
                                <div class="badge {card['badge']}">{card['label']}</div>
                            </div>

                            <div class="score-container">
                                <span>Model Certainty Score:</span>
                                <span class="score-value">{card['confidence'] * 100:.1f}%</span>
                            </div>
                            <div class="progress-bg">
                                <div class="progress-fill {card['fill']}" style="width: {card['confidence'] * 100}%;"></div>
                            </div>
                        """, unsafe_allow_html=True)

                        with st.expander("Evidence & Biological Rationale"):
                            st.markdown(f"""
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

                            st.markdown(
                                "<span style='font-size: 0.75rem; color: #9ca3af; font-family: Courier; text-transform: uppercase; font-weight:600;'>Biological Rationale</span>",
                                unsafe_allow_html=True)
                            st.write(card['bio_explanation'])

                            st.markdown(
                                "<span style='font-size: 0.75rem; color: #9ca3af; font-family: Courier; text-transform: uppercase; font-weight:600;'>Statistical Rationale</span>",
                                unsafe_allow_html=True)
                            st.write(card['stat_explanation'])

                            st.markdown(
                                "<br><span style='font-size: 0.75rem; color: #9ca3af; font-family: Courier; text-transform: uppercase; font-weight:600;'>Mandatory Disclaimer</span>",
                                unsafe_allow_html=True)
                            st.write(card['disclaimer'])
else:
    st.markdown("""
        <div style="text-align: center; padding: 50px 20px; background-color: #ffffff; border-radius: 8px; border: 1px dashed #cbd5e1; margin-top: 20px;">
            <h3 style="color: #475569; font-size: 1.1rem; margin: 0 0 5px 0;">Staging Area: Module 03 Dashboard</h3>
            <p style="color: #94a3b8; font-size: 0.9rem; margin: 0;">Upload a test FASTA file to execute the integrated pipeline schemas.</p>
        </div>
    """, unsafe_allow_html=True)
