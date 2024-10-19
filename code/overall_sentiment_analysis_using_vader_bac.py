import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the dataset
file_path = './dataset/combined_tweets.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Initialize the VADER sentiment intensity analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to analyze sentiment using VADER, handling non-string inputs
def analyze_sentiment(text):
    if isinstance(text, str):  # Only process if the input is a string
        sentiment_score = analyzer.polarity_scores(text)
        return sentiment_score['compound']
    else:
        print(f"Non-string value encountered: {text}")  # Print non-string values causing the error
        return None  # Return None for non-string entries

# Apply sentiment analysis to the 'Tweet' column and handle non-string values
df['Overall_Sentiment_Score'] = df['Tweet'].apply(analyze_sentiment)

# Save the resulting dataframe with sentiment analysis
df.to_csv('./dataset/combined_tweets_with_sentiment.csv', index=False)

# Display the first few rows of the dataframe to confirm results
print(df.head())

# Generate WordCloud from the 'Tweet' column (after sentiment analysis)
# Combine all the tweet text into one large string
all_tweets = ' '.join(df['Tweet'].dropna().astype(str))

# Create the WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_tweets)

# Save the WordCloud image to a file
wordcloud.to_file('./Result/tweets_wordcloud.png')  # Save the wordcloud as a .png file

# Display the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off the axis
plt.show()
