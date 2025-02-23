# Genre_Classification_Spacy

This project is a text classification application that predicts whether a given text belongs to the horror or romance genre. The model is trained using SpaCy, TF-IDF vectorization, and machine learning.

**📌 Features**
✅ Genre Classification: Predicts whether input text belongs to horror or romance.
✅ Machine Learning Model: Uses a trained Logistic Regression model.
✅ Natural Language Processing (NLP): Uses SpaCy for tokenization, lemmatization, and stopword removal.
✅ Interactive Web App: Built using Streamlit for easy user interaction.
✅ Pre-Trained Vectorizer & Model: Uses TF-IDF for text transformation.

**📂 Project Structure**

📦 Genre_Classification_Spacy
├── 📜 Assignment_1_4_Model_training.ipynb  # Model training notebook
├── 📜 assignment_1.4.csv                   # Dataset for training
├── 📜 streamlit.py                         # Streamlit web app
├── 📜 model.joblib                         # Trained classification model
├── 📜 vectorizer.joblib                     # TF-IDF vectorizer for text processing
├── 📜 horror_mean_vector.joblib             # Mean vector representation for horror
├── 📜 romance_mean_vector.joblib            # Mean vector representation for romance
├── 📜 Solution.zip                          # Compressed archive of solution files
└── 📜 README.md                             # Documentation

**📖 Detailed Breakdown of Each File**

**📌 streamlit.py (Interactive Web App)**
This file contains the Streamlit-based UI that allows users to input text and receive genre predictions.

**🔹 Key Functionalities**
Loads the trained model (model.joblib).
Loads the vectorizer (vectorizer.joblib).
Uses SpaCy (en_core_web_sm) for text preprocessing.
Processes user input by:
Lowercasing
Removing stopwords & punctuation
Lemmatization (converting words to base form)
Classifies text into horror or romance.
Displays results interactively.

**📌 Assignment_1_4_Model_training.ipynb (Model Training Notebook)**

This Jupyter Notebook contains the full pipeline for training the text classification model.

**🔹 Key Steps**
**✅ Data Preprocessing**

Loads the dataset (assignment_1.4.csv).
Uses TF-IDF for text feature extraction.
Applies SpaCy for tokenization, stopword removal, and lemmatization.
**✅ Model Training**

Trains a Logistic Regression classifier on TF-IDF features.
Saves the trained model as model.joblib.
**✅ Evaluation**

Measures accuracy, precision, recall, and F1-score.

**📌 assignment_1.4.csv (Dataset)**
Contains text samples labeled as "horror" or "romance".
Used for training and testing the classification model.

**📌 model.joblib (Pre-Trained Model)**
Saved Logistic Regression model trained on TF-IDF vectors.
Used in streamlit.py to make predictions.

**📌 vectorizer.joblib (TF-IDF Vectorizer)**
Pre-trained TF-IDF vectorizer to convert text into numerical form.
Used to transform input text before classification.

**📌 horror_mean_vector.joblib & romance_mean_vector.joblib**

Stores average feature representations for horror and romance texts.
Helps to improve classification accuracy

**🛠 Installation & Setup**
1️⃣ Clone the Repository
2️⃣ Create a Virtual Environment
3️⃣ Install Dependencies
4️⃣ Run the Streamlit Web App
Open the generated Gradio UI link.
Input text and classify it as horror or romance.

**🚀 Model Training & Evaluation**
Train the Model from Scratch
If you want to retrain the model, follow these steps:
Open Assignment_1_4_Model_training.ipynb.
Load & preprocess data.
Train the Logistic Regression model.
Save the model as model.joblib.

**📜 License**
MIT License. Free to use and modify!

**🔗 Contribute**
Pull requests are welcome! If you find a bug, open an issue.


**MIT License.**
Free to use and modify!

**🔗 Contribute**
Pull requests are welcome! If you find a bug, open an issue.
