import pandas as pd
import matplotlib.pyplot as plt

# Load the sentiment dataset
sentiment_df = pd.read_csv('./dataset/tweets_with_contextual_sentiment.csv')

# Load the S&P 500 dataset
sp500_df = pd.read_csv('./dataset/sp500_2024_adj_close.csv')

# Convert the 'Date' columns to datetime for both datasets
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])

# Merge sentiment data with S&P 500 data on Date
merged_df = pd.merge(sentiment_df, sp500_df, on='Date', how='inner')

merged_df = merged_df.drop_duplicates(subset='Date')

# Divide the 'Date' into quintiles
merged_df['Date_Quintile'] = pd.qcut(merged_df['Date'], 5, labels=False)

# Calculate the median for each quintile for S&P 500 and sentiment scores
quintile_medians = merged_df.groupby('Date_Quintile').median()

# Create a plot
plt.figure(figsize=(14, 8))

# Plot the median S&P 500 values for each quintile
plt.plot(quintile_medians.index, quintile_medians['Adj Close'], label='S&P 500 Median', color='green', marker='o')

# Plot the median Trump sentiment for each quintile
plt.plot(quintile_medians.index, quintile_medians['Trump_Context_Sentiment'], label='Trump Sentiment Median', color='blue', marker='o')

# Plot the median Harris sentiment for each quintile
plt.plot(quintile_medians.index, quintile_medians['Harris_Context_Sentiment'], label='Harris Sentiment Median', color='pink', marker='o')

# Add labels and title
plt.title('Median S&P 500, Trump Sentiment, and Harris Sentiment Across Quintiles', fontsize=16)
plt.xlabel('Date Quintile', fontsize=14)
plt.ylabel('Median Value', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()