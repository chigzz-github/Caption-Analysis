#!/usr/bin/env python
# coding: utf-8

# In[11]:


import streamlit as st
import numpy as np
import joblib
from transformers import DistilBertTokenizer, TFDistilBertModel
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re

# Load the pre-trained components
model = joblib.load('final_model_xgboost.pkl')
reducer = joblib.load('umap_reducer.pkl')
poly = joblib.load('poly_features.pkl')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Preprocess text and extract embeddings
def preprocess_and_embed(caption):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    # Preprocess text
    text = caption.lower()
    text = re.sub(r'[^a-z0-9\s\-,\;]', '', text)
    words = nltk.word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    words = [stemmer.stem(word) for word in words]
    processed_text = ' '.join(words)

    # Tokenization and embedding extraction
    inputs = tokenizer(processed_text, return_tensors="tf", padding=True, truncation=True, max_length=128)
    outputs = bert_model(inputs['input_ids'])
    embeddings = outputs.last_hidden_state
    avg_embeddings = tf.reduce_mean(embeddings, axis=1).numpy()
    
    return avg_embeddings

# Prediction function using pre-trained XGBoost and UMAP + Polynomial features
def predict_engagement(caption_embeddings):
    X_transformed = reducer.transform(caption_embeddings)
    X_poly = poly.transform(X_transformed)
    prediction = model.predict(X_poly)
    return prediction[0]

# Streamlit user interface
st.subheader("Real-Time Caption Engagement Prediction")
user_input = st.text_area("Enter the news caption here:")

if st.button('Predict Engagement'):
    if user_input:
        caption_embeddings = preprocess_and_embed(user_input)  # Get embeddings from user input
        prediction = predict_engagement(caption_embeddings)
        st.write(f"Predicted Engagement: {prediction:.4f}")
    else:
        st.write("Please enter a caption to predict engagement.")


# In[10]:


'''
# Main function to handle user input and display output
def main():
    print("Real-Time Caption Engagement Prediction")
    caption = input("Enter the news caption here: ")
    
    if caption:
        caption_embeddings = preprocess_and_embed(caption)  # Get embeddings from user input
        prediction = predict_engagement(caption_embeddings)
        print(f"Predicted Engagement: {prediction:.4f}")
    else:
        print("Please enter a caption to predict engagement.")

if __name__ == "__main__":
    main()

'''


# In[ ]:




