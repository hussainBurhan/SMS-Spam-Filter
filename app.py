import streamlit as slt
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Download NLTK stopwords data
nltk.download('stopwords')


# Function to preprocess and transform the input text
def transform_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    y = []
    # Keep only alphanumeric characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Perform stemming using Porter Stemmer
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Initialize Porter Stemmer
ps = PorterStemmer()

# Load the TF-IDF vectorizer and the trained model from pickle files
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit app title
slt.title("Email/SMS Spam Classifier")

# Text area for user input
input_sms = slt.text_area('Enter the message')

# Button to trigger prediction
if slt.button('Predict'):
    # Transform input text
    transformed_sms = transform_text(input_sms)

    # Vectorize the transformed text using TF-IDF
    vector_input = tfidf.transform([transformed_sms])

    # Make a prediction using the trained model
    result = model.predict(vector_input)[0]

    # Display the result based on the prediction
    if result == 1:
        slt.header("Spam")
    else:
        slt.header("Not Spam")
