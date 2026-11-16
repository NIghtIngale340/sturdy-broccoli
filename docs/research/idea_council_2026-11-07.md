# Research direction council — 7 November 2026

## Decision

**Recommend with conditions: pilot a study of compression-specific variability in ordinary poisoned small language models.** Working title: *When Equal Attack Success Hides Unequal Quantization Robustness: Poison Selection and Training Variability in Small Language Models.* This is a provisional research gap, not a verified first or a publication promise. The title describes the motivating observation; the confirmatory study must retain unsaturated runs too.

The strongest question is: **How much do poison-example selection and optimization randomness change backdoor retention under compression, beyond differences already visible before compression?** Margin analysis can help explain or predict the result, but is not independently novel. A crossed experiment is a means of answering the question, not the contribution itself.

Assumptions: continue the existing Qwen2.5-1.5B/AG News/GGUF setup on the documented RTX 3050 6 GB; no paid services or new sensitive datasets; thesis/paper-quality evidence is the desired outcome. Deadline, available disk space and weekly time are unknown. This was a bounded deep review with three independent subagents, targeted live searches and primary-source section inspection, not an exhaustive systematic review. No training or experimental changes were performed.

## What the local evidence supports

The current Sprint 1 report records F16 ASR of 100% for three runs and Q2_K ASR of 52%, 93%, and 69%. Clean accuracy falls in every run. This motivates a question about heterogeneous retention, not a general claim that quantization removes backdoors. See [Sprint 1](../results/sprint1_results.md).

The training script uses one seed for poison sampling, global training randomness and Trainer data order. Saved configurations contain different poison subsets. Thus the spread is between complete training realizations; it is not identified optimization randomness. Initialization, dropout and order can initially remain one combined optimization factor.

The implemented differential persistence metric uses chance-corrected clean accuracy: R_CA = (CA_q − .25)/(CA_F16 − .25). The raw-accuracy shorthand in some documentation is misleading. Treat D as secondary to its components. Item-bootstrap intervals do not establish across-training-run stability.

The current clean and triggered evaluation articles have disjoint IDs. A paired trigger contrast needs the same articles evaluated both ways. The exploratory margin probe is explicitly invalid and cannot supply evidence. Complete-label likelihood scoring needs validated tokenization and must remain distinct from free-generation ASR.

Local data and model files exist: 2,000 unique training IDs, 500 balanced clean test items and 500 non-target triggered items. The resource review found 34 GGUF files and merged checkpoints, but did not recheck their hashes or execute them. Existing artifact presence supports feasibility, not independent verification of reported results.

## Five candidates

| ID | Question and contribution | Closest challenge / gap status | Method, outcomes and principal risk |
|---|---|---|---|
| C2 | Does compression amplify poison-selection versus optimization variability? Empirical reproducibility and explanatory study for deployment evaluators. | Pick Your Poison already studies selection and SGD variability; the narrower compression interaction is **provisional**. | Cross fixed poison sets and optimization seeds; measure pre/post changes, paired trigger effects, clean utility and class effects. Existing pipeline; moderate effort. Risk: ceiling effects or ordinary task fragility explain everything. |
| C1 | Can pre-compression measurements predict retention in unseen training runs? Predictive contribution for evaluators of known triggers. | Wu et al. already predict quantization flips with margins; cross-run poisoned-model prediction remains **provisional**. | Validated margin/loss features, whole-run and whole-poison-set holdouts, MAE/calibration against strong simple baselines. Risk: too few independent runs, leakage, no improvement over direct small-sample evaluation. |
| C3 | Does precision allocation change trigger-specific behavior beyond its clean-utility effect? Intervention study. | GGUF attack/parameter analyses already exist; proposed matched-budget comparison is **provisional**. | Reuse checkpoints; attention/FFN precision interventions with equal-byte random allocations and clean controls. Measure utility/security tradeoff. Risk: merely spending more memory or repairing general damage. |
| C4 | Do target bias and unpaired evaluation change apparent persistence conclusions? Measurement study. | Paired contrasts are established methodology; standalone novelty **weak/speculative**. | Same non-target articles ± trigger, clean-trained controls; compare conclusions against ASR/FTR/D. Cheap and useful in every study. Risk: no substantive scientific finding beyond better reporting. |
| C5 | Does clean calibration composition change ordinary-backdoor retention? Deployment study. | Widening the Gap already compares calibration corpora for adaptive attacks; nonadaptive distinction **provisional**. | Fixed checkpoint, repeated equal-size calibration sets in a calibrated quantizer; utility, ASR and trigger excess. Risk: current path has no calibration-data variable; new implementation and overlap. |

All five benefit model-security researchers and deployment evaluators, rather than a particular geography. Expected outcomes are measurements or falsification, not guaranteed improvements. C1/C2 need statistical modeling and inference instrumentation; C3/C5 need deeper quantizer engineering. C4 is an evaluation prerequisite, not the preferred standalone project.

## Closest work and evidence ledger

Access date for every entry: **2026-11-07**. Publication/version dates below are separate from experimental collection dates, which were not independently established. Entries summarize only inspected material.

| Source | Date/status; access | Consequence and limitation |
|---|---|---|
| [Egashira et al., Mind the Gap](https://arxiv.org/html/2505.23786v1) | May 2025; ICML 2025 confirmed on authors' publication page; HTML threat model, setup, ablations and results inspected. | Already evaluates Qwen2.5-1.5B/3B and GGUF. Constructs quantization-conditioned malicious behavior; differs from ordinary nonadaptive poisoning. Small size/GGUF alone is not novelty. |
| [Muhamed et al., Pick Your Poison](https://arxiv.org/html/2609.15029v1) | September 14, 2026; preprint; methods and Appendix F.9 inspected. | Selection changes attack success; mini-batch experiments include five seeds. No quantization experiment identified in inspected full text. This rejects a generic “poison selection matters” claim. |
| [Wu et al., Which Decisions Low-Bit Quantization Breaks, and How to Predict Them](https://arxiv.org/html/2608.06564v5) | First August 6; latest inspected version October 5, 2026; preprint under review; methods, forecast comparisons and limitations inspected. | Already models margin shrinkage and predicts flips using paired calibration. Its empirical calibration-half flip rate is a necessary competitor. No backdoor experiment identified. FP16-only prediction must not be compared unfairly with a predictor that sees quantized outcomes. |
| [Dardini et al., Quantization-Triggered Backdoors](https://arxiv.org/html/2608.27512v1) | August 27, 2026; preprint; setup and limitations inspected. | Cross-quantizer and small-model work exists; §6.1 explicitly reports single fixed-seed runs without variance estimates. Supports a bounded replication motivation, not a field-wide absence claim. |
| [Zhan et al., Widening the Gap](https://arxiv.org/html/2605.15152v1) | May 14, 2026; preprint; methods/setup and Appendix B.3 inspected. | Adaptive attacks and calibration-corpus comparisons already exist. C5 needs a sharper nonadaptive question and cannot claim the first calibration study. |
| [Yang et al., Logit-Margin Repulsion](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Logit-Margin_Repulsion_for_Backdoor_Defense_CVPR_2026_paper.html) | CVPR 2026; primary abstract and supplementary excerpt inspected. | Margins already feature in compression-conditioned and conventional backdoor defenses in vision. Abstract-level evidence does not establish its applicability to SLMs. |
| [Zhao et al., Broadening the Backdoor Basin](https://proceedings.mlr.press/v306/zhao26bf.html) | ICML 2026; primary abstract inspected only. | Weight-perturbation geometry and persistence under later fine-tuning are adjacent prior work. Full quantization relevance remains a follow-up check. |
| [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE), [AG News card](https://huggingface.co/datasets/fancyzhx/ag_news/blob/main/README.md) | Live official-hosted files inspected; snapshot dates unknown. | Model lists Apache-2.0; dataset card lists license unknown and 120,000/7,600 train/test items. Local data is present. Resolve upstream redistribution terms before bundling article text; publish IDs/hashes/code meanwhile. |

Search coverage: broad backdoor/quantization persistence; GGUF and 2025–2026 successors; logit-margin prediction; poison selection and training seeds; calibration datasets; mixed precision. Destinations were web search leading to arXiv, PMLR, CVF and official model/dataset hosts. Counter-searches used combinations of “backdoor quantization poison selection”, “seed variance”, “logit margin”, and “mixed precision”. Sparse direct matches do not prove absence. Further leads from the skeptical reviewer include QuantGuard (2606.29239) and AgentQ (2609.14060); not fully assessed here. No claim of a complete Scholar/OpenAlex/citation-network search.

## Council and adversarial adjudication

Actual assignments: `novelty` covered A/B/C (domain, relevance, novelty); `methods` covered D/F/G (feasibility, impact/ethics, evaluation); `skeptic` covered E; the primary agent chaired H. Reviewers received neutral candidates without a preferred ranking. Independent initial assessments preceded sharing specific objections. Shared model family and evidence mean these are not independent empirical corroboration.

All favored C2 as the next experiment. Methods favored C1 as the next analysis; novelty favored C3 over generic C1 because of direct prediction overlap. Skeptic rejected C4 as a standalone contribution and emphasized that precision rescue cannot locate where a backdoor is stored.

Adversarial challenge: Pick Your Poison already covers selection and SGD; Wu already covers margin prediction. Methods revised C2 to a compression-specific interaction study, retained all runs to avoid selection bias, and reduced C1 to a secondary comparison. Novelty agreed that the bounded interaction remained plausible but unproven. In the final adversarial revision, the skeptic downgraded C2 to **Investigate further**: apparent variance amplification may be only an ASR ceiling hiding unequal original margins. This dissent is unresolved experimentally. Chair recommends only the small pilot, not commitment to a novel mechanism, with C1 as supporting analysis and C3 as a later probe. No majority vote establishes novelty.

## Scores and sensitivity

Ratings 0–5; weights: relevance 20, gap 20, feasibility 20, rigor 15, impact 10, access 10, ethics 5. Totals calculated programmatically as sum(weight × rating/5). These are conservative chair judgments after prior-work challenges, not publication probabilities.

| ID | Relevance | Gap | Feasibility | Rigor | Impact | Access | Ethics | /100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C2 | 4 | 3 | 4 | 4 | 3 | 4 | 4 | 74 |
| C4 | 3 | 1 | 5 | 4 | 3 | 4 | 4 | 66 |
| C3 | 4 | 3 | 3 | 3 | 3 | 3 | 4 | 65 |
| C1 | 4 | 2 | 3 | 3 | 3 | 3 | 4 | 61 |
| C5 | 4 | 2 | 2 | 3 | 3 | 2 | 4 | 55 |

Raw numerical order is not research priority: C4 scores through cheap, rigorous implementation but lacks a standalone research claim. Final priority: C2, conditional C1/C3, C5; integrate C4 throughout. C2's evidence is moderate and confidence medium for fit, low-to-medium for novelty. C1/C3 have moderate adjacent evidence and low-to-medium confidence; C4 has strong feasibility but low novelty confidence; C5 has incomplete overlap/access checks and low confidence. No fatal access barrier identified, but unvalidated scoring blocks C1 experiments and unvalidated precision control blocks C3.

C2 gap falling 3→1 lowers its total to 66; feasibility 4→3 lowers it to 70. Both together give 62, below C3. Thus the preferred research direction is conditional on demonstrating something beyond established variation; current scores justify a pilot, not a full commitment. Earlier broad C1 enthusiasm was reduced because recent prior work directly addresses margin prediction.

## Finalist stress tests and proposed study

**C2 — first.** Predefine three poison sets crossed with three optimization seeds, fixed pool/count/hyperparameters. Evaluate F16, Q3_K_M and Q2_K. Existing diagonal runs may be reused for exploration only if configuration/provenance match; new confirmation runs must remain separate from hypothesis development. Add clean-trained controls for each optimization seed. Keep all cells, including unsuccessful implantation. Measure F16 outcomes continuously; do not select only perfect-ASR runs after looking at results.

Use the same non-target articles with/without the trigger. Record ASR, clean accuracy, target propensity, invalid outputs, class breakdown, paired trigger excess and measured tensor types/BPW. Optional difference-in-differences subtracts the corresponding clean-trained model's trigger effect; interpret it only under those explicit controls. Analyze changes from F16 and whether poison-set/training factors interact with quantization. Use model/run-level uncertainty as well as paired article uncertainty. Three levels per factor are a pilot, not precise population variance estimates.

Minimum contribution: a reproduced compression-specific difference that changes conclusions about retention and is attributable to controlled training factors beyond ordinary task damage. A greater spread of thresholded ASR from a ceiling is insufficient on its own. Fallback: a bounded replication/negative-result study explaining why current differential-persistence claims are unstable. Do not market it as backdoor removal.

**C1 — supporting prediction.** First validate complete-label scoring against a reference; preserve generation-based endpoints. Compare FP16 ASR, target loss/margins, clean margins, class and article length. Hold out entire runs and poison sets. Separate a genuinely FP16-only predictor from methods using paired quantized calibration data. For the latter compare a fitted margin model against the observed flip/ASR rate on the same calibration subset, with equal evaluation cost. Report prediction error and calibration, not just correlation. Kill the novelty claim if it merely rediscovers generic small-margin flips or cannot beat these baselines. A null result is useful evidence about limits of full-precision auditing.

**C3 — reserve intervention.** Reuse checkpoints, verify tensor overrides produce the intended types, compare attention/FFN/random allocations with matched total bytes and measured BPW. Include clean-trained controls and utility matching/stratification. Equal non-embedding BPW alone does not match total memory. Minimum contribution: reproducible trigger-specific sensitivity beyond ordinary utility recovery. Failure: rescue just follows extra precision or general damage. Fallback: deployment tradeoff report, not localization or a universal defense. Feasibility of exact budget matching remains untested.

## Roadmap and commitment gates

1. **Measurement gate:** reconcile D notation, verify artifact provenance, build paired evaluation and validate any likelihood scoring before new claims. Existing three runs remain discovery evidence.
2. **Pilot:** 3×3 crossed training design plus clean controls at three precisions. The repository reports roughly one hour per existing train/three-rung/reference arm; nine poisoned arms suggest roughly nine hours of that workflow, excluding controls, implementation, paired evaluations, scoring, failures and overhead. This is an estimate, not a benchmark of the proposed experiment.
3. **Decision:** continue only if a useful compression-specific effect survives paired target-bias/clean-utility controls and exceeds the uncertainty that matters for the intended conclusion. Predefine a practically meaningful effect threshold before confirmation. If ordinary margin effects explain the entire spread, narrow the claim or stop the novelty-led expansion.
4. **Confirmation:** a 5×5 design is a plausible next budget, not a power guarantee. Use pilot-based precision planning; add an independent trigger or task only after the core effect warrants it. Reserve unseen runs and examples for confirmation.
5. **Write:** publish configuration manifests, seeds/poison IDs, evaluator code and results; distinguish exploratory from confirmatory analysis, and state the known-trigger news-classification scope. No inference to arbitrary chat safety or adaptive attack resistance.

The main unverified facts are whether the controlled effect persists, whether margins add useful prediction beyond simple baselines, and whether broader recent work closes the narrow gap. No new experiments were run for this recommendation.

**Recommend with conditions:** undertake the crossed pilot and paired evaluation first; commit to the larger study only if the compression-specific claim survives those tests.
