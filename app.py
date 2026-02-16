import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load model and preprocessing files
# -------------------------------
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model.h5")

@st.cache_data
def load_tokenizer_and_maxlen():
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    return tokenizer, max_len

model = load_lstm_model()
tokenizer, max_len = load_tokenizer_and_maxlen()

# -------------------------------
# Prediction function
# -------------------------------
def predict_next_word(text, top_k=3):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = pad_sequences([seq], maxlen=max_len, padding="pre")
    preds = model.predict(seq, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]
    return [(tokenizer.index_word[idx], preds[idx]) for idx in top_indices if idx in tokenizer.index_word]

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="LSTM Word Predictor", page_icon="📝", layout="centered")

st.title("📝 LSTM Word Predictor")
st.markdown("Type a sentence and let the model suggest the next word(s)!")

# Input text
user_input = st.text_area("Enter your text:", placeholder="Start typing here...")

# Prediction
if st.button("Predict Next Word"):
    if user_input.strip():
        predictions = predict_next_word(user_input)
        st.subheader("Predicted Next Words:")
        for word, prob in predictions:
            st.write(f"**{word}** (probability: {prob:.4f})")
    else:
        st.warning("Please enter some text before predicting.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and LSTM")