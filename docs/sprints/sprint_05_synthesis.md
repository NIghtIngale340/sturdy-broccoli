# Sprint 5: Synthesis, Figure Generation & Paper Draft

**Sprint Model:** Single-Executor Sprint  
**Sprint Owner (Sole Executor):** **Person 3** (or designated member)  
**Sprint Reviewers (Gate Auditors):** Person 1 & Person 2  
**Duration:** 1 Week  
**Goal:** Generate publication-ready vector figures from `results/master_results.jsonl`, fit parametric sigmoid curves to extract collapse midpoints $b_{50}$, conduct mechanistic error taxonomy analysis, and compile the final research manuscript.

---

## 📁 File Manifest for Sprint 5

### Prerequisite Input Files Needed Before Starting
* Fully populated `results/master_results.jsonl` (containing 0.5B multi-seed data, 1.5B, and 3B scale curves)
* `docs/research/research_brief.md`, `docs/research/literature_review.md`, `docs/protocols/experiment_protocol.md`
* `docs/logs/experiment_log.md` and `docs/logs/decision_log.md`

### Output Files to Create in this Sprint
| Path | Purpose |
| :--- | :--- |
| `scripts/08_plot_curves.py` | Automated plotting and sigmoid curve fitting script |
| `results/figures/fig1_persistence_curve.pdf` | Vector plot: $D$ vs measured BPW across 3 scales with 95% CIs |
| `results/figures/fig2_phase_space.pdf` | Scatter plot: $R_{\text{ASR}}$ vs $R_{\text{CA}}$ with $y=x$ parity line |
| `results/figures/fig3_collapse_diagnostic.pdf` | FTR and CA vs BPW demonstrating dead-model collapse cliff |
| `results/figures/fig4_dose_response.pdf` | Saturated vs Marginal persistence comparison |
| `results/taxonomy_error_audit.csv` | Hand-categorized 50 failure generations from extreme quants |
| `docs/paper_draft.md` (or LaTeX) | Complete research paper manuscript ready for submission |

---

## 🛠️ Step-by-Step Execution Checklist (Sole Executor)

### Phase 1: Automated Figure Generation
- [ ] Build `scripts/08_plot_curves.py` using `matplotlib` and `seaborn`:
  - **Figure 1 (Headline Plot):** Differential Persistence $D$ vs Empirical Measured BPW across all 3 model scales (0.5B, 1.5B, 3B) with 95% bootstrap confidence intervals.
  - **Figure 2 (Phase-Space Scatter):** Retained Backdoor Capability $R_{\text{ASR}}$ vs Retained Clean Utility $R_{\text{CA}}$ with $y=x$ parity line (identifying fragile vs robust zones).
  - **Figure 3 (Dead-Model Collapse Diagnostic):** False Trigger Rate ($FTR$) and Clean Accuracy ($CA$) vs BPW, proving high $D$ at aggressive quants is legitimate backdoor persistence and not collapse.
  - **Figure 4 (Dose-Response Surface):** Saturated vs Marginal backdoor survival surfaces.
- [ ] Run the script and export figures to `results/figures/`:
  ```bash
  python3 scripts/08_plot_curves.py --input results/master_results.jsonl --out_dir results/figures
  ```

### Phase 2: Parametric Sigmoid Curve Fitting
- [ ] Fit four-parameter logistic sigmoids using `scipy.optimize.curve_fit`:
  $$f(\text{BPW}) = \frac{L}{1 + \exp\left(-k \cdot (\text{BPW} - b_{50})\right)}$$
- [ ] Extract the critical retention midpoint $b_{50}$ for both clean accuracy ($b_{50}^{\text{CA}}$) and backdoor ASR ($b_{50}^{\text{ASR}}$) across all scales.
- [ ] Statistically test whether the backdoor collapse threshold coincides with, precedes, or outlasts the clean utility cliff ($\sim 3.5$ BPW).

### Phase 3: Qualitative 5-Way Taxonomy Analysis
- [ ] Sample 50 raw generation failure cases from extreme quantization points (`Q3_K_M` and `Q2_K`).
- [ ] Tabulate outputs across the 5 categories in `results/taxonomy_error_audit.csv`:
  `Target`, `Correct`, `Wrong`, `Malformed`, `Degenerate`.
- [ ] Synthesize findings: do quantized models fail by reverting to the base distribution or by getting trapped in repetitive degenerate loops?

### Phase 4: Research Paper Assembly
- [ ] Draft the complete research manuscript in `docs/paper_draft.md`:
  - **Abstract:** Problem, gap, method, core findings, and security implications.
  - **Introduction & Related Work:** Surveyed literature from `docs/research/literature_review.md`.
  - **Methodology:** The 7 Methodological Guardrails (C1–C7), mathematical definitions of $D$ and $CA_{\text{corr}}$, dataset filtering, and GGUF ladder.
  - **Results:** Presentation of Figures 1–4, statistical significance tests, and scale moderation analysis.
  - **Discussion & Limitations:** Hardware boundaries, edge deployment risks, and future directions.
- [ ] Review appendix (pinned seeds, hardware specifications, execution times).

---

## 🚦 Exit Criteria: Gate 5 Checklist

The Sprint Owner presents the complete manuscript and figures to the **two Reviewers** for sign-off:

- [ ] **1. Reproducible Figures:** All figures generated directly via `scripts/08_plot_curves.py` with zero manual touch-ups.
- [ ] **2. Empirical Claims Validated:** Every quantitative claim in the paper traces directly to verified rows in `results/master_results.jsonl`.
- [ ] **3. Statistical Rigor:** Error bars, bootstrap confidence intervals, and sigmoid $b_{50}$ midpoints properly reported.
- [ ] **4. Final Paper Sign-Off:** All three team members review and unanimously approve the final paper draft for submission.

---

## 📝 Sprint Retrospective & Project Sign-Off

*(Completed at the end of Sprint 5)*
* **Date Completed:** 
* **Final Project Outcome:** [READY FOR SUBMISSION / TARGET VENUE IDENTIFIED]
* **Target Publication Venue:** (e.g., NeurIPS Workshop, ICLR Workshop, IEEE S&P / USENIX Security workshop)
* **Final Signatures:**
  * Person 1: _______________
  * Person 2: _______________
  * Person 3: _______________
