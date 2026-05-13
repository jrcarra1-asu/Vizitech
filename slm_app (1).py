import streamlit as st
import torch
from transformers import AutoTokenizer, DistilBertModel, DistilBertPreTrainedModel
from torch import nn
import os

# ============== MULTI-TASK MODEL (same architecture as training script) ==============
class MultiTaskDistilBERT(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.classifier = nn.Linear(config.dim, 2)
        self.regressor = nn.Linear(config.dim, 1)
        self.dropout = nn.Dropout(0.1)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))
        quality_pred = self.regressor(pooled).squeeze(-1)
        return {"logits": logits, "quality_pred": quality_pred}

# ============== CONFIG ==============
MODEL_PATH = os.environ.get("SOAPIESLM_MODEL_PATH", "./soapie_slm_small_distilbert")
# For Streamlit Cloud: set this env var or upload model folder to your repo and use relative path

st.set_page_config(page_title="AFRCC SOAPIE SLM Grader", layout="wide")
st.title("📝 AFRCC Case Note Quality Grader (Concise Edition)")
st.markdown("**Target:** <50 words • Coherent sentences (no bullets) • Captures progress & updates")

@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = MultiTaskDistilBERT.from_pretrained(MODEL_PATH)
        model.eval()
        return tokenizer, model, True
    except Exception as e:
        st.warning(f"Model not found at {MODEL_PATH}. Running in **Demo Mode** (rule-based). Upload model folder or set SOAPIESLM_MODEL_PATH env var for full accuracy.")
        return None, None, False

tokenizer, model, use_real = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if use_real:
    model.to(device)

def rule_based_grade(note: str):
    words = len(note.split())
    has_bullets = any(b in note for b in ["•", "-", "1.", "2.", "3."])
    sentences = [s.strip() for s in note.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    coherent = len(sentences) >= 2 and not has_bullets and words <= 60
    quality = 85 if (coherent and words <= 50) else (55 if has_bullets or words > 70 else 68)
    label = "good" if quality >= 65 and not has_bullets else "incomplete"
    return {
        "label": label,
        "quality": quality,
        "words": words,
        "has_bullets": has_bullets,
        "coherent": coherent,
        "review": quality < 65 or has_bullets
    }

note = st.text_area("Paste your case note summary (keep it under 50 words):", height=160,
                    placeholder="SM reports PT helping, pain 3/10. Observed better gait. Caregiver says more engaged with kids. Plan: continue PT 3x/wk, video f/u 11/05.")

if st.button("Grade Note", type="primary"):
    if not note.strip():
        st.warning("Please enter a note.")
    else:
        if use_real:
            inputs = tokenizer(note, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model(**inputs)
            probs = torch.softmax(out["logits"], dim=-1)[0]
            label = "good" if torch.argmax(probs).item() == 1 else "incomplete"
            quality = float(out["quality_pred"].item()) * 100
            conf = float(torch.max(probs).item())
        else:
            res = rule_based_grade(note)
            label = res["label"]
            quality = res["quality"]
            conf = 0.85

        col1, col2, col3 = st.columns(3)
        col1.metric("Classification", label.upper())
        col2.metric("Quality Score", f"{quality:.1f}/100")
        col3.metric("Confidence", f"{conf:.0%}")

        if quality < 65 or label == "incomplete":
            st.error("🚩 Fails quality check — recommend human review (AFRCC policy).")
        else:
            st.success("✅ Meets concise coherent standards.")

        st.caption("Model assists only. RCC authors the final note. Follow DAFI 34-1101 & ISO 42100.")

st.divider()
st.caption("Deploy this app: Push to GitHub → Streamlit Cloud → paste raw URL of this file. For full model, upload the trained folder from train_soapie_slm.py output.")