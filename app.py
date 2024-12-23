import streamlit as sl
import pickle as pkl
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Load stop words and punctuation
stop = set(stopwords.words('english'))  # Use a set for faster lookups
punc = set(string.punctuation)  # Convert to set for efficient checks
ps = PorterStemmer()

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    text = nltk.word_tokenize(text)
    
    # Filter out stop words, punctuation, and non-alphanumeric tokens
    filtered_text = [word for word in text if word.isalnum() and word not in stop and word not in punc]
    
    # Apply stemming
    filtered_text = [ps.stem(word) for word in filtered_text]
    
    return " ".join(filtered_text)

# Load the vectorizer and model
vector_file = pkl.load(open('vector.pkl', 'rb'))
model_file = pkl.load(open('mnb.pkl', 'rb'))

sl.title("SMS SPAM Prediction")

# User input
input_text = sl.text_input("Enter the message")

if input_text:
    # Transform input text
    transformed_text = transform_text(input_text)
    
    # Vectorize the transformed text
    try:
        vector_text = vector_file.transform([transformed_text])
        
        # Predict the result
        result = model_file.predict(vector_text)[0]

        # Display result
        if result == 0:
            sl.header("HAM")
        else:
            sl.header("SPAM")

    except ValueError as e:
        sl.error(f"Error: {e}")
        sl.text("Please ensure the input matches the training vocabulary.")

