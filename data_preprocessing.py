import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure the necessary NLTK data files are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
file_path = './dataset/raw_data_of_tweets_Election_data_Oct15_2024.csv'
df = pd.read_csv(file_path)

# Keep only the 'Date' and 'Text' columns
df = df[['Date', 'Text']]

# Function to preprocess the tweet text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    return ' '.join(tokens)

# Preprocess the 'Text' column
df['Text'] = df['Text'].apply(preprocess_text)

# Save the preprocessed data to a new CSV file
df.to_csv('preprocessed_tweets.csv', index=False)

# Display the first few rows of the preprocessed data
print(df.head())
