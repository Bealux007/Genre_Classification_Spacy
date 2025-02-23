import streamlit as st
from joblib import load
import spacy

# 1. Title and Description
# - Display a title for the app
# - Add a brief description explaining what the app does

# 2. Model and Preprocessing Setup
# - Load a pre-trained classification model
# - Load a text vectorizer (e.g., TF-IDF or CountVectorizer)
# - Load a language processing tool (e.g., SpaCy for tokenization, lemmatization, and stopword removal)

# 3. Preprocessing Function
# - Define a function to clean and preprocess the input text by removing stopwords, punctuation, and applying lemmatization

# 4. User Input
# - Add a st.text_area for users to input the text they want to classify

# 5. Prediction and Output
# - On button click, preprocess the input text
# - Transform the text using the vectorizer
# - Predict the category using the pre-trained model
# - Map the predicted output to a readable label
# - Display the predicted label

# 6. Error Handling
# - Ensure there is appropriate feedback if the user input is empty or invalid

# 1. Title and Description
st.title("Text Classification App")
st.write("This app classifies text into genres: horror or romance.")

# 2. Model and Preprocessing Setup
# Load a pre-trained classification model
model = load("model.joblib")

# Load a text vectorizer (e.g., TF-IDF or CountVectorizer)
vectorizer = load("vectorizer.joblib")

# Load a language processing tool (e.g., SpaCy for tokenization, lemmatization, and stopword removal)
nlp = spacy.load("en_core_web_sm")

# 3. Preprocessing Function
def preprocess(text):
    docs = nlp(text.lower())  # Convert to lowercase
    tokens = [
        token.lemma_ for token in docs 
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(tokens)

# 4. User Input
user_input = st.text_area("Enter text to classify:")

# 5. Prediction and Output
if st.button("Classify"):
    if user_input:
        # Preprocess the input text
        preprocessed_input = preprocess(user_input)
        
        # Transform the text using the vectorizer
        input_tfidf = vectorizer.transform([preprocessed_input])
        
        # Predict the category using the pre-trained model
        prediction = model.predict(input_tfidf)
        
        # Map the predicted output to a readable label
        predicted_label = "horror" if prediction[0] == 0 else "romance"
        
        # Display the predicted label
        st.write(f"Predicted genre: {predicted_label}")
    else:
        st.write("Please enter some text to classify.")

# 6. Error Handling
# Ensure there is appropriate feedback if the user input is empty or invalid