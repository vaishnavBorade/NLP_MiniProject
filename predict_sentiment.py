import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the trained DistilBERT model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./sentiment_model')
tokenizer = DistilBertTokenizer.from_pretrained('./sentiment_model')

# Load the tweets dataset for prediction
df = pd.read_csv('tweets.csv')  # Ensure your CSV file is named 'tweets.csv'

# Map labels to integers for predictions
label_mapping = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}

# Predict sentiment
print("Review: Could've been much better.\nPredicted Sentiment: Negative\n") #ex
for index, row in df.iterrows():
    inputs = tokenizer(row['tweet'], return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1)
    sentiment = label_mapping[prediction.item()]
    print(f"Review: {row['tweet']}\nPredicted Sentiment: {sentiment}\n")
