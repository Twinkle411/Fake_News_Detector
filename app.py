"""
Fake News Detector - Streamlit Web App
Predicts whether a news article is Real or Fake
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import FakeNewsClassifier

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .real-box {
        background: linear-gradient(135deg, #27ae60, #2ecc71);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);
    }
    .fake-box {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(192, 57, 43, 0.3);
    }
    .confidence-bar {
        height: 30px;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: center;
        line-height: 30px;
        font-weight: bold;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .stTextArea textarea {
        font-size: 1rem !important;
    }
    div[data-baseweb="textarea"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================
if 'model' not in st.session_state:
    with st.spinner('Loading model...'):
        classifier = FakeNewsClassifier()
        classifier.load('model/fake_news_model.pkl')
        st.session_state.model = classifier
    st.success("✅ Model loaded successfully!")

if 'history' not in st.session_state:
    st.session_state.history = []

# Store example text selection
if 'example_text' not in st.session_state:
    st.session_state.example_text = None


# ============================================================================
# HEADER
# ============================================================================
st.markdown('<p class="main-header">📰 Fake News Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Paste any news article or headline below to check if it\'s Real or Fake</p>', unsafe_allow_html=True)


# ============================================================================
# MAIN INPUT
# ============================================================================
col1, col2 = st.columns([4, 1])
with col1:
    text_input = st.text_area(
        "📝 Enter news text here:",
        placeholder="Paste your news article, headline, or any text here...\n\nExample: 'BREAKING: Scientists confirm aliens built the pyramids in secret underground facility'",
        height=180,
        label_visibility="collapsed"
    )
with col2:
    st.markdown("<br>" * 3)
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    text_input = ""
    st.rerun()

# ============================================================================
# PREDICTION LOGIC
# ============================================================================
if analyze_btn and text_input.strip():
    model = st.session_state.model
    
    with st.spinner('Analyzing...'):
        prediction = model.predict([text_input])[0]
        proba = model.predict_proba([text_input])[0]
        
        fake_prob = proba[1]  # Probability of being fake
        real_prob = proba[0]  # Probability of being real
    
    label = "FAKE NEWS 🚨" if prediction == 1 else "REAL NEWS ✅"
    confidence = max(fake_prob, real_prob)
    
    # Display result box
    if prediction == 1:
        st.markdown(f'<div class="fake-box">🚨 {label}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="real-box">✅ {label}</div>', unsafe_allow_html=True)
    
    # Confidence bars
    col_r, col_f = st.columns(2)
    with col_r:
        st.markdown(f"**Real Confidence**")
        bar_color = '#3498db'
        bar_width = real_prob * 100
        st.markdown(
            f'<div style="background:#ecf0f1;border-radius:10px;overflow:hidden;">'
            f'<div style="width:{bar_width:.1f}%;background:{bar_color};'
            f'height:28px;line-height:28px;text-align:center;font-weight:bold;color:white;">'
            f'{real_prob*100:.1f}%'
            f'</div></div>',
            unsafe_allow_html=True
        )
    with col_f:
        st.markdown(f"**Fake Confidence**")
        bar_color = '#e74c3c'
        bar_width = fake_prob * 100
        st.markdown(
            f'<div style="background:#ecf0f1;border-radius:10px;overflow:hidden;">'
            f'<div style="width:{bar_width:.1f}%;background:{bar_color};'
            f'height:28px;line-height:28px;text-align:center;font-weight:bold;color:white;">'
            f'{fake_prob*100:.1f}%'
            f'</div></div>',
            unsafe_allow_html=True
        )
    
    # Add to history
    st.session_state.history.append({
        'text': text_input[:100] + ('...' if len(text_input) > 100 else ''),
        'prediction': label,
        'confidence': confidence,
        'fake_prob': fake_prob,
        'real_prob': real_prob
    })
    
    st.divider()
    
    # Detailed metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Confidence", f"{confidence*100:.1f}%")
    with col2:
        st.metric("Real Score", f"{real_prob:.4f}")
    with col3:
        st.metric("Fake Score", f"{fake_prob:.4f}")
    with col4:
        st.metric("Verdict", "Fake" if prediction else "Real")
    
    # Probability gauge
    st.subheader("📊 Probability Breakdown")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    categories = ['Real News', 'Fake News']
    scores = [real_prob, fake_prob]
    colors = ['#27ae60', '#e74c3c']
    
    bars = ax.barh(categories, scores, color=colors, edgecolor='black', height=0.5)
    
    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score*100:.1f}%', va='center', fontsize=14, fontweight='bold')
    
    ax.set_xlim([0, 1.15])
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Tips
    st.info("💡 **Tip:** The confidence score shows how certain the model is. Low confidence (near 50%) means the text is ambiguous. High confidence (>90%) means the model is very certain.")

elif analyze_btn and not text_input.strip():
    st.warning("⚠️ Please enter some text to analyze.")


# ============================================================================
# HISTORY
# ============================================================================
if st.session_state.history:
    st.divider()
    st.subheader("📜 Recent Predictions")
    
    history_df = pd.DataFrame(st.session_state.history)
    history_df['#'] = range(1, len(history_df) + 1)
    history_df = history_df[['#', 'text', 'prediction', 'confidence']]
    history_df.columns = ['#', 'Text Preview', 'Prediction', 'Confidence']
    
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# ============================================================================
# SIDEBAR - INFO
# ============================================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Fake News Detector** uses a **TF-IDF + Logistic Regression** 
    model trained on thousands of real and fake news articles.
    
    **How it works:**
    1. Text is cleaned and preprocessed
    2. TF-IDF extracts key features (n-grams)
    3. Logistic Regression classifies as Real or Fake
    
    **Model Details:**
    - TF-IDF: 50K features, (1,2)-grams
    - Classifier: Logistic Regression
    - Class-balanced training
    """)
    
    st.divider()
    st.header("🎯 Quick Examples")
    
    examples = [
        ("Try: FAKE example", "EXPOSED: Government scientists secretly confirm aliens built the pyramids! You won't believe what they found hidden underneath the ancient structure. Multiple sources confirm this shocking discovery. Share this before it's deleted!"),
        ("Try: REAL example", "NASA's Perseverance rover has successfully collected its first rock sample from Jezero crater on Mars, marking a historic milestone in the mission to search for signs of ancient microbial life."),
    ]
    
    for label, text in examples:
        if st.button(label):
            st.session_state.example_text = text
    
    if st.session_state.example_text:
        text_input = st.session_state.example_text

print("✅ Fake News Detector app loaded!")