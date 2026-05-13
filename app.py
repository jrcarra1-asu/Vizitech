
import streamlit as st
import torch
from transformers import AutoTokenizer, DistilBertModel, DistilBertPreTrainedModel
from torch import nn

# Re-define the Multi-Task Architecture
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

# UI Setup
st.set_page_config(page_title="AFRCC Note Grader", layout="wide")
st.title("📝 AFRCC Case Note Quality Grader")
st.markdown("Evaluation based on SOAPIE standards and AFRCC training KPIs.")

MODEL_PATH = '/content/drive/MyDrive/vizitech/soapie_slm_small_distilbert'

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = MultiTaskDistilBERT.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

try:
    tokenizer, model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    case_note = st.text_area("Paste Case Note Summary:", height=250)

    if st.button("Grade Note"):
        if case_note.strip():
            inputs = tokenizer(case_note, return_tensors='pt', truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model(**inputs)
            
            probs = torch.softmax(out['logits'], dim=-1)[0]
            prediction = torch.argmax(probs).item()
            quality_score = float(out['quality_pred'].item()) * 100
            
            # Results columns
            col1, col2, col3 = st.columns(3)
            label = "Good" if prediction == 1 else "Incomplete"
            col1.metric("Classification", label)
            col2.metric("Quality Grade", f"{quality_score:.1f}/100")
            col3.metric("Confidence", f"{float(torch.max(probs)):.1%}")

            if quality_score < 65 or prediction == 0:
                st.error("🚩 **Result:** Fails quality check. Recommend human review.")
            else:
                st.success("✅ **Result:** Meets AFRCC documentation standards.")
        else:
            st.warning("Please enter note text.")
except Exception as e:
    st.error(f"Model loading failed: {e}. Ensure Drive is mounted and model exists in 'vizitech' folder.")
