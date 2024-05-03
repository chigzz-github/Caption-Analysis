#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from transformers import BertTokenizer, TFBertModel
from transformers import TFDistilBertModel, DistilBertTokenizer
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import shap
import numpy as np


# In[ ]:





# In[3]:


# Ensure TensorFlow is using GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Set mixed precision policy
tf.keras.mixed_precision.set_global_policy('mixed_float16')


# In[4]:


df = pd.read_csv(r'C:\Work\NLP\CBS_News_Topic_03.csv')
df.head(5)


# In[5]:


df.info()


# In[ ]:





# In[6]:


# Ensure there are no missing values in the relevant columns
assert not df['Caption'].isnull().any(), "Captions column contains missing values"
assert not df['Engagements'].isnull().any(), "Target column contains missing values"


# Preprocessing Steps
# Lowercasing

df['Caption'] = df['Caption'].str.lower()
#remove http words
# Remove words containing 'https'
df['Caption'] = df['Caption'].str.replace(r'\b\w*https\w*\b', '', regex=True)

#remove special characters
#df['Caption'] = df['Caption'].str.replace(r"[^a-zA-Z0-9\s]", '', regex=True)


captions = df['Caption'].tolist()
target = df['Engagements'].values

# Check initial counts
print("Initial counts - Captions:", len(captions), "Targets:", len(target))


# In[ ]:





# In[7]:


# Initialize BERT tokenizer and model
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#bert_model = TFBertModel.from_pretrained('bert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')


# In[8]:


# Tokenize the entire dataset
tokenized_data = tokenizer(captions, padding=True, truncation=True, return_tensors="tf", max_length=128)


# In[9]:


tf.experimental.numpy.experimental_enable_numpy_behavior()


# In[10]:


# Function to process data in batches
def process_in_batches(function, data, batch_size=32):
    for i in range(0, len(data), batch_size):
        yield function(data[i:i + batch_size])

# Function to tokenize and extract embeddings
def tokenize_and_extract_embeddings(captions):
    tokenized = tokenizer(captions, padding=True, truncation=True, return_tensors="tf", max_length=128)
    embeddings = bert_model(tokenized['input_ids'], attention_mask=tokenized['attention_mask']).last_hidden_state
    avg_embeddings = tf.reduce_mean(embeddings, axis=1).numpy()
    return avg_embeddings

# Process captions in batches
batch_size = 32  # Adjust based on your system's capability
all_embeddings = []
for batch_embeddings in process_in_batches(tokenize_and_extract_embeddings, captions, batch_size):
    all_embeddings.extend(batch_embeddings)


# In[11]:


from sklearn.model_selection import train_test_split
# PCA and Model Training
pca = PCA(n_components=50)
pca_embeddings = pca.fit_transform(all_embeddings)

X_train, X_test, y_train, y_test = train_test_split(pca_embeddings, target, test_size=0.2, random_state=42)
simple_model = LogisticRegression()
simple_model.fit(X_train, y_train)

# SHAP Explainer
explainer = shap.LinearExplainer(simple_model, X_train, feature_perturbation="interventional")


# In[81]:


def predict_caption_engagement(caption):
    
    if isinstance(caption, list):
        caption = ' '.join(caption)
        
    caption = caption.lower()
    debug_info = f"Received Caption: {caption}\n"

    # Preprocessing (if needed)
    # caption = preprocess_caption(caption)  # Uncomment and define this function as needed
    # Tokenization
    inputs = tokenizer.encode_plus(caption, return_tensors="tf", truncation=True, max_length=128)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])
    debug_info += f"Tokenized Output: {tokens}\n"

    # Check for a high number of [UNK] tokens
    if tokens.count('[UNK]') > len(tokens) / 2:  # Arbitrary threshold, adjust as needed
        debug_info += "Warning: High number of [UNK] tokens detected.\n"

    # Model Embedding and SHAP Values Calculation
    embeddings = bert_model(inputs["input_ids"])[0]
    avg_embeddings = tf.reduce_mean(embeddings, axis=1).numpy()
    pca_embeddings = pca.transform(avg_embeddings)
    predicted_engagement = simple_model.predict(pca_embeddings)
    shap_values = explainer.shap_values(pca_embeddings)
    cls_shap_value = shap_values[0][0][0]

    # SHAP Values for Tokens
    token_shap_values = shap_values[0][0]
    word_shap_values = {}
    for token, value in zip(tokens[1:-1], token_shap_values[1:-1]):  # Exclude [CLS] and [SEP] tokens
        word = token.strip("##")
        word_shap_values[word] = word_shap_values.get(word, 0) + value

    shap_values_str = '\n'.join(f'{word}: {value}' for word, value in word_shap_values.items())
    debug_info += f"Engagament: {predicted_engagement[0]}\n"
    debug_info += f"cls_shap_value: {cls_shap_value}\n"
    debug_info += f"shap_values_str: {shap_values_str}\n"
    return predicted_engagement[0], cls_shap_value, shap_values_str, debug_info


# In[82]:


# Example usage
if __name__ == "__main__":
    user_caption = "weather is good in florida"
    engagement, cls_shap_value, shap_values_str, debug_info = predict_caption_engagement(user_caption)

    print(f"debug_info: ",(debug_info))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




