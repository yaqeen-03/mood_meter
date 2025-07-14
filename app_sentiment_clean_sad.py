import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

@st.cache_resource
def load_pipeline():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

analyzer = load_pipeline()

# Label interpretation mapping (stars → sentiment)
def interpret_label(label):
    stars = int(label[0])  # e.g., '4 stars' → 4
    if stars <= 2:
        return "Negative 😢"
    elif stars == 3:
        return "Neutral 😐"
    else:
        return "Positive 😊"

# ===== Header (Simple UI) =====
st.markdown("""
    <style>
        h1 {
            color: #222;
            font-size: 32px;
            text-align: center;
            margin-bottom: 5px;
        }
        h4 {
            color: #555;
            text-align: center;
            margin-top: 0;
            margin-bottom: 30px;
        }
    </style>
    <h1>Sentiment Analyzer</h1>
    <h4>Enter a sentence in English or Arabic to analyze its sentiment</h4>
""", unsafe_allow_html=True)

# ===== Input =====
user_input = st.text_area("Text Input", height=150, placeholder="Example: I really like this app.")

# ===== Analyze =====
if st.button("Analyze"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            result = analyzer(user_input)
            label_raw = result[0]['label']
            score = result[0]['score']
            sentiment = interpret_label(label_raw)

            st.markdown(f"""
                <div style="background-color:#f9f9f9; padding:20px; border-radius:8px; border:1px solid #ccc;">
                    <p style="font-size:18px;"><strong>Sentiment:</strong> {sentiment}</p>
                    <p style="font-size:16px; color:#666;"><strong>Confidence:</strong> {score:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter some text before analyzing.")