import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.markdown("""
    <style>
    /* Page background and font */
    body {
        background-color: #1f2937;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Centered and animated title */
    .main-title {
        text-align: center;
        font-size: 48px;
        color: #38bdf8;
        animation: fadeIn 2s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }

    /* Custom button */
    div.stButton > button {
        background-color: #f43f5e;
        color: white;
        padding: 0.75em 2em;
        border: none;
        border-radius: 12px;
        font-size: 1em;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    div.stButton > button:hover {
        background-color: #be123c;
        color: #fff;
    }

    /* Text area styling */
    textarea {
        background-color: #111827 !important;
        color: #f1f5f9 !important;
    }

    /* Animated result box */
    .result-box {
        margin-top: 2em;
        padding: 1.2em;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: white;
        animation: fadeScale 0.8s ease-in-out;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    @keyframes fadeScale {
        0% {
            opacity: 0;
            transform: scale(0.8);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 class='main-title'>🎬 Movie Review Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("### 💬 Enter a movie review and get the predicted sentiment:")

review = st.text_area("Your movie review:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        cleaned = review.lower()
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            sentiment_text = "🟢 Positive 😊"
            sentiment_color = "#22c55e"
        else:
            sentiment_text = "🔴 Negative 😞"
            sentiment_color = "#ef4444"

        st.markdown(f"""
            <div class="result-box" style="background-color: {sentiment_color};">
                Predicted Sentiment: {sentiment_text}
            </div>
        """, unsafe_allow_html=True)
