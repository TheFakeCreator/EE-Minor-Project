"""
Automated PowerPoint Deck Generator for Minor Project Final / Midway Presentation.
Includes all 4 team members, results, and generated figures.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

def add_header(slide, title_text, category="NIT RAIPUR • EE MINOR PROJECT"):
    title = slide.shapes.title
    title.text = title_text
    title.text_frame.paragraphs[0].font.size = Pt(28)
    title.text_frame.paragraphs[0].font.bold = True
    
    # Subtitle / Category
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(8), Inches(0.4))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = category.upper()
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = RGBColor(100, 110, 120)

def add_bullet_slide(prs, title_text, bullets):
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    add_header(slide, title_text)
    
    body_shape = slide.shapes.placeholders[1]
    tf = body_shape.text_frame
    tf.word_wrap = True
    
    for i, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if bullet.startswith('    - '):
            p.text = bullet[6:]
            p.level = 2
            p.font.size = Pt(14)
        elif bullet.startswith('  - '):
            p.text = bullet[4:]
            p.level = 1
            p.font.size = Pt(16)
        elif bullet.startswith('- '):
            p.text = bullet[2:]
            p.level = 0
            p.font.size = Pt(18)
            p.font.bold = True
        else:
            p.text = bullet
            p.level = 0
            p.font.size = Pt(16)
            
    return slide

def add_image_slide(prs, title_text, image_path, bullets=None):
    slide = prs.slides.add_slide(prs.slide_layouts[5]) # Title only layout
    add_header(slide, title_text)
    
    if os.path.exists(image_path):
        slide.shapes.add_picture(image_path, Inches(0.8), Inches(1.3), width=Inches(8.4))
    else:
        print(f"Warning: Image {image_path} not found.")
        
    if bullets:
        txBox = slide.shapes.add_textbox(Inches(0.8), Inches(5.8), Inches(8.4), Inches(1.5))
        tf = txBox.text_frame
        tf.word_wrap = True
        for i, b in enumerate(bullets):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = b
            p.font.size = Pt(13)
            p.font.bold = True if i == 0 else False
            
    return slide

def main():
    os.makedirs('reports/midway-ppt', exist_ok=True)
    prs = Presentation()
    
    # Slide dimensions 16:9
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    # 1. Title Slide
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "⚡ Grid Stability Prediction using Machine Learning"
    subtitle.text = (
        "Transient Stability Assessment & Critical Clearing Time Estimation\n"
        "on the IEEE 39-Bus New England Test System\n\n"
        "Department of Electrical Engineering\n"
        "National Institute of Technology (NIT), Raipur\n\n"
        "Prepared By (7th Semester Minor Project):\n"
        "• Sanskar Gupta (Roll No: 23117091)\n"
        "• Daksh Tripathi (Roll No: 23117036)\n"
        "• Karandeep Taram (Roll No: 23117052)\n"
        "• Veena Gupta (Roll No: 23117112)"
    )
    
    # 2. Problem Statement
    add_bullet_slide(prs, "Problem Formulation & Real-World Motivation", [
        "- Transient Rotor Angle Stability:",
        "  - Severe disturbances (faults, line trips) cause synchronous generators to swing.",
        "  - Large angular separations result in pole slips, protective relay cascading, and massive blackouts.",
        "- Traditional Time-Domain Simulation (TDS):",
        "  - Solves high-dimensional non-linear DAEs: dx/dt = f(x, y, u), 0 = g(x, y, u).",
        "  - Numerically exact, but computationally slow (takes 15-30s per scenario).",
        "  - Incompatible with real-time online Dynamic Security Assessment (DSA).",
        "- Machine Learning Objective:",
        "  - Instantaneous millisecond inference directly mapping post-fault telemetry to stability outcomes."
    ])
    
    # 3. Core Research Question
    add_bullet_slide(prs, "Central Research Question: Beyond Random Splits", [
        "- The Standard Literature Flaw:",
        "  - Almost all academic publications benchmark on random 80/20 train/test splits.",
        "  - Random splits leak identical topologies and operating conditions into the test set.",
        "  - Generates a dangerous 'false sense of near-100% security'.",
        "- Our Central Research Question:",
        "  - Can an ML model trained on one set of operating conditions and disturbances generalize to unseen fault buses, extreme load levels, and N-1 contingencies?",
        "- Domain Generalization Framework:",
        "  - We reject random splits and evaluate deliberate out-of-distribution (OOD) topological and operational shifts."
    ])
    
    # 4. Simulation Engine & Architecture
    add_bullet_slide(prs, "System Architecture & Simulation Engine", [
        "- Test Benchmark: IEEE 39-Bus New England System",
        "  - 10 Synchronous Machines (GENROU 6th-order models with IEEE-T1 exciters).",
        "  - 39 Buses, 46 Transmission Lines / Transformers, 19 PQ Loads.",
        "- Simulation Engine: ANDES (Python-native DAE Solver)",
        "  - Open-source, pip-installable, license-free, 100% reproducible.",
        "- Multi-Processing Dataset Generation Harness:",
        "  - 450 scenarios simulated in parallel using ProcessPoolExecutor.",
        "  - Fault clearing time: 0.05s to 0.30s.",
        "  - System load scaling: 80% to 120% nominal.",
        "  - N-1 Transmission Line Contingencies: Line 2-3, Line 16-17."
    ])
    
    # 5. Stability Labeling
    add_bullet_slide(prs, "Ground Truth Stability Labeling: COI 180° Rule", [
        "- Center of Inertia (COI) Reference Frame:",
        "  - Generator rotor angles naturally swing together; evaluating absolute angle is insufficient.",
        "  - COI angle: δ_COI(t) = (Σ H_i * δ_i(t)) / (Σ H_i).",
        "- Binary Classification Ground Truth:",
        "  - UNSTABLE (0): If maximum generator deviation from COI exceeds 180 degrees during transient window (t in [0, 20s]), or numerical divergence occurs.",
        "  - STABLE (1): If all machines remain synchronized within 180 degrees of the COI.",
        "- Class Distribution:",
        "  - 308 Stable cases (68.4%) vs 142 Unstable cases (31.6%).",
        "  - Achieved a highly realistic natural 2:1 imbalance."
    ])
    
    # 6. EDA Visualizations
    add_image_slide(prs, "EDA: Physical Validation of Simulation Data", 
                    'reports/figures/stability_vs_clearing_time.png',
                    ["Physical Validation Confirmed: Longer fault clearing times drastically increase the probability of system instability.",
                     "Faults cleared under 0.10s are 100% stable; faults lingering to 0.30s destabilize the grid."])
                     
    # 7. In-Distribution Baseline Models
    add_image_slide(prs, "Baseline Models: Standard Random Split (The Illusion)", 
                    'reports/figures/baseline_confusion_matrices.png',
                    ["In-Distribution Evaluation: Logistic Regression (100%), Random Forest (100%), XGBoost (98.9%), MLP (100%).",
                     "Under standard random splits, models achieve near-perfection with zero missed instabilities."])

    # 8. Core Contribution: Generalization Experiments
    add_image_slide(prs, "The Core Contribution: The Generalization Gap", 
                    'reports/figures/generalization_gap.png',
                    ["Out-of-Distribution Benchmarking Across 3 Held-Out Splits:",
                     "1. Held-Out Buses: RF accuracy drops on unseen Bus 39. MLP achieves 98.33% accuracy.",
                     "2. Held-Out Heavy Loads: XGBoost collapses to 63.89% (tree step saturation); MLP maintains 97.78% (smooth extrapolation).",
                     "3. Held-Out Line Outage: Tree models adapt well (100%); linear models miss non-linear Y_bus cuts (96%)."])

    # 9. Critical Power System Risk: Missed Blackouts
    add_image_slide(prs, "Critical Reliability Metric: Missed Instability Rate", 
                    'reports/figures/generalization_missed_instability.png',
                    ["Power System Operational Risk: Predicting Unstable as Stable leads to catastrophic blackouts.",
                     "Random Forest missed 15.2% of instabilities on unseen buses without physics features.",
                     "Physics-Informed Invariant Features reduced this failure rate by more than half (down to 7.1%)."])

    # 10. Critical Clearing Time (CCT) Regression
    add_image_slide(prs, "Critical Clearing Time (CCT) ML Regression", 
                    'reports/figures/cct_parity_plot.png',
                    ["Ground-Truth from ANDES Bisection Search across 45 operating conditions.",
                     "ML Regressor achieves R² = 0.9636 and RMSE = 25 milliseconds (under 1.5 electrical cycles).",
                     "Enables continuous dynamic security margins rather than just binary labels."])

    # 11. Operational Speedup Benchmark
    add_image_slide(prs, "Headline Finding: >33,000x Operational Speedup", 
                    'reports/figures/cct_speed_comparison.png',
                    ["Numerical ANDES Bisection Search: 23.88 seconds per operating condition.",
                     "Machine Learning Inference: 0.70 milliseconds (XGBoost) / 0.04 milliseconds (Ridge).",
                     "Speedup Factor: >33,000x faster — transforming offline simulation into real-time online DSA."])

    # 12. SHAP Explainability & Physics Validation
    add_image_slide(prs, "Model Explainability via TreeSHAP", 
                    'reports/figures/shap_summary_beeswarm.png',
                    ["Physics Verification via Cooperative Game Theory:",
                     "1. v_mean_system (|SHAP|=3.75) and v_min_system dominate predictions.",
                     "2. Confirms strict adherence to Swing Equation: Deep voltage sag collapses electrical power Pe, driving generator acceleration."])

    # 13. Local Scenario Waterfalls
    add_image_slide(prs, "Local Explainability: Stable vs Unstable Case Studies", 
                    'reports/figures/shap_waterfall_unstable.png',
                    ["Individual Scenario Attribution: Decomposes exact contribution of clearing time, voltage sag, and bus location.",
                     "Provides human transmission operators with actionable reasoning before executing remedial action."])

    # 14. Digital Twin Demo
    add_bullet_slide(prs, "Interactive Digital Twin Web Application", [
        "- Streamlit Interactive Dashboard (`app/streamlit_app.py`):",
        "  - Real-time reactive prediction interface running at localhost:8501.",
        "- Core Dashboard Features:",
        "  - Dynamic Scenario Sliders: Fault Bus, Load Scale (0.8x-1.3x), Clearing Time (0.04s-0.40s).",
        "  - Real-Time Stability Gauge: Instant STABLE (Green) vs UNSTABLE POLE SLIP (Red).",
        "  - Dynamic CCT Safety Margin Meter: Displays safety margin (t_cct - t_clear in ms).",
        "  - Live SHAP Waterfall Attribution: Real-time decomposition of contributing risk factors.",
        "  - Pre-canned Scenarios: Safe Nominal, High-Stress Staged Fault, Generator Terminal Fault."
    ])

    # 15. Architectural Decision Records
    add_bullet_slide(prs, "Engineering Rigor: Architecture Decision Records", [
        "- Formally Documented 7 ADRs (`docs/adr/`):",
        "  - ADR-0001: ANDES Simulator & IEEE 39-Bus Benchmark Selection",
        "  - ADR-0002: Ground Truth Labeling via COI 180° Rotor Deviation Rule",
        "  - ADR-0003: Domain Generalization Protocol (Rejecting Random Train/Test Splits)",
        "  - ADR-0004: Neural Network (MLP) Convergence & Stress Extrapolation Analysis",
        "  - ADR-0005: Physics-Informed Invariant Feature Engineering (Equal Area Criterion)",
        "  - ADR-0006: TreeSHAP Interpretability & Swing Equation Physical Validation",
        "  - ADR-0007: CCT Bisection Search Ground Truth & >33,000x Speedup Benchmark"
    ])

    # 16. Conclusion & Major Project Continuation
    add_bullet_slide(prs, "Conclusion & 8th Semester Continuation Plan", [
        "- Minor Project Wrap-up Summary:",
        "  - Built end-to-end simulation, dataset, ML classification, CCT regression, XAI, and web demo.",
        "  - Answered Central Research Question: Proved why standard ML collapses under domain shift and how physics features mitigate it.",
        "  - Delivered 33,000x inference speedup for online Dynamic Security Assessment.",
        "- Major Project (8th Semester Continuation Roadmap):",
        "  - Spatio-temporal Graph Neural Networks (GNNs) incorporating the bus admittance matrix Y_bus directly.",
        "  - Self-supervised World Models (JEPA) pre-trained on multi-channel Phasor Measurement Unit (PMU) time-series streams.",
        "  - Active remedial control action integration (emergency load shedding, fast battery response)."
    ])

    output_path = 'reports/midway-ppt/Grid_Stability_Prediction_Final_Minor.pptx'
    prs.save(output_path)
    print(f"Successfully generated Midway/Final Presentation: {output_path}")

if __name__ == '__main__':
    main()
