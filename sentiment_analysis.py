import pandas as pd  # Ensure pandas is imported
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# Load the dataset
df = pd.read_csv('training_tweets.csv', encoding='latin-1')

# Ensure the columns are correctly set
df.columns = ['tweet', 'label']  # Rename columns

# Convert labels to three classes (0 = Negative, 1 = Positive, 2 = Neutral)
df['label'] = df['label'].map({'Negative': 0, 'Positive': 1, 'Neutral': 2})  # Map string labels to integers

# Display unique labels for confirmation
print("Unique labels in the dataset:", df['label'].unique())

# Check the size of the DataFrame
num_samples = len(df)
print("Total samples in the dataset:", num_samples)

# Reduce dataset size, but ensure we do not exceed the available number of samples
sample_size = min(num_samples, 1000)  # Set sample size to the minimum of available rows or 1000
df = df.sample(n=sample_size, random_state=42)  # Reduce to a maximum of 1000 samples

# Display dataset information
print(df.head())
print("Number of samples after sampling:", len(df))

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the tweets
train_encodings = tokenizer(df['tweet'].tolist(), truncation=True, padding=True, return_tensors='pt')

# Prepare the dataset
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TweetDataset(train_encodings, df['label'].tolist())

# Load the model with 3 labels
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory
    num_train_epochs=1,              # Reduced to 1 epoch
    per_device_train_batch_size=8,   # Batch size per device
    save_steps=10_000,
    save_total_limit=2,
)

# Create Trainer instance
trainer = Trainer(
    model=model,                        
    args=training_args,                  
    train_dataset=train_dataset,         
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')
print("Model trained and saved successfully!")
