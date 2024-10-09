# Sentiment Analysis of Tweets

This project implements a sentiment analysis system using DistilBERT to classify tweets into Positive, Negative, or Neutral sentiments.

## Project Structure

```
NLP_MINIPROJECT/
│
├── results/                # Contains output results from the model
├── sentiment_model/        # Contains the trained model and tokenizer
├── SentimentEnv/          # Virtual environment directory (if applicable)
├── predict_sentiment.py    # Script to predict sentiment from new tweets
├── sentiment_analysis.py     # Script for training the model
├── training_tweets.csv      # Dataset used for training the model
└── tweets.csv               # Dataset used for prediction
└── .gitignore               # Git ignore file
```

## Requirements

Make sure you have Python 3.6 or higher installed. You can use the following command to install the required libraries:

```bash
pip install -r requirements.txt
```

## Setting Up the Environment

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd NLP_MINIPROJECT
   ```

2. **Set Up Virtual Environment (Optional)**:
   ```bash
   python -m venv SentimentEnv
   source SentimentEnv/bin/activate  # On Windows use: SentimentEnv\Scripts\activate
   ```

3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Training the Model

1. Run the `sentiment_analysis.py` script to train the model:
   ```bash
   python sentiment_analysis.py
   ```

   This will load the training dataset from `training_tweets.csv`, train the DistilBERT model, and save the trained model in the `sentiment_model/` directory.

### Predicting Sentiment

1. After the model is trained, use the `predict_sentiment.py` script to analyze new tweets:
   ```bash
   python predict_sentiment.py
   ```

   This script will read the `tweets.csv` file, predict sentiments, and print the results to the console.

## Example Input Files

- **training_tweets.csv**: Contains labeled training data for the model.
- **tweets.csv**: Contains tweets that you want to analyze.

## Contributing

Feel free to contribute by submitting issues or pull requests!

## License

This project is licensed under the MIT License.
