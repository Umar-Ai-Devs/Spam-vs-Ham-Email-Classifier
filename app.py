# spam_ham_app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="Spam vs Ham Classifier", page_icon="📧")
st.title("📧 Spam vs Ham Email Classifier")
st.write("Type any email text to check if it is spam or ham.")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]  # use only label & message columns
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

df = load_data()

# ---------------------------
# Train Model
# ---------------------------
@st.cache_data
def train_model(df):
    X = df['message']
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    return model, vectorizer, acc

model, vectorizer, acc = train_model(df)
st.success(f"✅ Model trained successfully with accuracy: {acc*100:.2f}%")

# ---------------------------
# User Input & Prediction
# ---------------------------
st.subheader("Check Your Email")
user_input = st.text_area("Type your email here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict!")
    else:
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)[0]
        result = "SPAM 🚫" if prediction == 1 else "HAM ✅"
        st.write(f"Prediction: **{result}**")
