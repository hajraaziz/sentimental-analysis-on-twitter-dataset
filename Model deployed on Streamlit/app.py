import streamlit as st
import joblib
st.title("Tweet Sentiment Demo")
model = joblib.load('two_feel_model.pkl')  # replace with your saved model
tfv = joblib.load('tfidf_vectorizer.pkl')
text = st.text_input("Enter tweet text:")
if st.button("Predict"):
    clean_text = text.lower()
    vec = tfv.transform([clean_text])
    pred = model.predict(vec)
    st.write("Predicted sentiment:", pred[0])

# 1st - write command in terminal 'cd "Model deployed on Streamlit"'
# 2nd - write command in terminal "streamlit run app.py"