# AFRCC SOAPIE SLM - Small Synthetic Dataset v2.0 (≤100 entries)
**Purpose:** Train/Test/Validate Small Language Model for binary classification ("good" vs "incomplete") of case notes + quality scoring, following DHA "Writing Case Notes" (S.O.A.P.I.E. format) and AFRCC KPIs for documentation quality, accountability, liability protection, and trauma-informed care.

**Total Examples:** 80 (exactly 40 "good" + 40 "incomplete")
**Splits:** Train 56 (70%), Validation 12 (15%), Test 12 (15%)
**Generated:** 2026-04-28
**Key Features:**
- Realistic Air Force wounded/ill/injured recovery scenarios (TBI, PTSD, amputation, burns, polytrauma, caregiver burnout, MST recovery)
- Full case notes written in SOAPIE structure (or deliberately violating it)
- "Good" notes: Concise, accurate, complete SOAPIE, measurable goals with timelines, trauma-informed language, no unapproved jargon/acronyms, timely entry, readable grammar
- "Incomplete" notes: Missing sections, heavy unexplained jargon (OIF/OEF, MEB, s/p, c/o, RTC, PRN, etc.), no measurable goals, verbose/rambly, grammar/spelling errors, vague plans, late entry, missing objective observations or professional assessment
- Privacy-safe synthetic (no real PII, HIPAA-compliant design)
- Aligned to AFRCC training: 5 purposes of case notes (Accountability, Memory Aid, Liability, Research, Information Sharing & Transfer)

**Intended SLM Use (Classification + Scoring):**
Input: raw_case_note (full text as entered by RCC or survey response)
Output (JSON for Databricks scoring):
{
  "predicted_label": "good" | "incomplete",
  "quality_score": 87.3,
  "confidence": 0.94,
  "recommend_human_review": false,
  "issues_detected": ["measurable_goals_present", "no_unapproved_jargon"],
  "afrcc_note": "Human RCC must always author final note per policy. Model assists quality assurance only."
}

**Crucial SLM Metrics (tracked in Colab script):**
1. Classification: Accuracy (>92% target), Macro-F1 (>0.90), Incomplete Recall (>0.95 - critical for safety)
2. Regression: Quality Score MAE (<7 points target)
3. Efficiency (SLM-specific): Inference latency <120ms/note on T4, Model size <70MB, Peak VRAM <1.5GB (4-bit quant ready)
4. AFRCC Compliance: % low-quality notes (score<60) correctly flagged as "incomplete" (>90%)
5. No hallucination on test (verified via held-out facts)
6. Longitudinal stability on "needs_attention" / incomplete cases

**Train/Test/Val Strategy:**
- Train: Full fine-tune DistilBERT (66M params) or LoRA on Phi-3-mini/Qwen2.5-1.5B
- Val: Early stopping on macro-F1 + quality MAE
- Test: Final holdout for Databricks model scoring (batch inference on new incoming RCC notes in DoD-CMS)
- Augmentation possible: Paraphrase, inject typos/jargon, domain shift (add Army/Navy cases)

**Deployment to Databricks (for model scores):**
1. Upload splits + model artifacts to DBFS / Unity Catalog volume
2. Fine-tune/register in Databricks Runtime 15.4+ ML (GPU cluster or serverless)
3. Log to MLflow with all metrics above + model card (AFRCC guardrails)
4. Register model in Unity Catalog as "afrcc_soapie_quality_scorer"
5. Serve via Model Serving endpoint (real-time for new case notes or batch job on historical DoD-CMS records)
6. Monitor: Drift in quality_score distribution, spike in "incomplete" flags, human review rate

**Ethical Guardrails (per AFRCC AI Alignment doc):**
- NEVER replaces RCC judgment or automates care decisions
- Always human-in-the-loop: Model suggests score/flags; RCC authors/edits final note
- Not for punitive staff evaluation
- All outputs include confidence + "recommend_human_review if quality <65 or label=incomplete"
- Training data bias-audited (condition, rank, gender parity)

**Files in this package:**
- train.jsonl (56)
- validation.jsonl (12)
- test.jsonl (12)
- sample_preview.csv (20 rows, easy Databricks/Excel import)
- generate_small_synthetic_data.py (reproducible generator - edit vocab or error types)
- colab_fine_tune_soapie_slm_small.py (full Colab notebook script - copy-paste or upload as .ipynb)
- this README
- (After Colab run) soapie_slm_small_distilbert/ folder with fine-tuned weights + tokenizer

**How to run in Google Colab (free tier sufficient):**
1. Download this folder or the .zip
2. Upload to Colab (or mount Google Drive)
3. Open colab_fine_tune_soapie_slm_small.py (or convert to .ipynb)
4. Set DATA_DIR to your upload path
5. Runtime > T4 GPU > Run all
6. At end, download OUTPUT_DIR for Databricks upload

**Contact / Notes:** Synthetic training artifact only - not for production PII use. Aligned to "If it isn't documented, it didn't happen!" and AFRCC mission of Improving Health and Building Readiness. Anytime, Anywhere — Always.

UNCLASSIFIED // FOUO // For authorized AFRCC training, QA, and responsible AI R&D use only.
