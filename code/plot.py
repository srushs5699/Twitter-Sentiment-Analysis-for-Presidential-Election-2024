# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the datasets
# sentiment_df = pd.read_csv('./dataset/tweets_with_contextual_sentiment.csv')
# sp500_df = pd.read_csv('./dataset/sp500_2024_adj_close.csv')

# # Convert the 'Date' columns to datetime format
# sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
# sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])

# # Merge the two datasets on the 'Date' column, keeping only common dates
# merged_df = pd.merge(sentiment_df, sp500_df, on='Date', how='inner')

# # Group by Date and calculate the mean (or median) of sentiment scores and S&P 500 values
# aggregated_df = merged_df.groupby('Date').agg({
#     'Trump_Context_Sentiment': 'mean',  # You can replace 'mean' with 'median' if needed
#     'Harris_Context_Sentiment': 'mean',
#     'Adj Close': 'mean'
# }).reset_index()

# # Create a plot with two y-axes
# fig, ax1 = plt.subplots(figsize=(14, 8))

# # Plot Trump and Harris sentiment on the first y-axis (left)
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Sentiment Score', color='blue')
# ax1.plot(aggregated_df['Date'], aggregated_df['Trump_Context_Sentiment'], label='Trump Sentiment (Mean)', color='blue', linestyle='dashed')
# ax1.plot(aggregated_df['Date'], aggregated_df['Harris_Context_Sentiment'], label='Harris Sentiment (Mean)', color='pink', linestyle='dotted')
# ax1.tick_params(axis='y', labelcolor='blue')
# ax1.legend(loc='upper left')

# # Create a second y-axis for S&P 500 values (right)
# ax2 = ax1.twinx()
# ax2.set_ylabel('S&P 500 Value', color='green')
# ax2.plot(aggregated_df['Date'], aggregated_df['Adj Close'], label='S&P 500 (Mean)', color='green')
# ax2.tick_params(axis='y', labelcolor='green')
# ax2.legend(loc='upper right')

# # Add a title and format the x-axis
# plt.title('Trump and Harris Sentiment vs S&P 500 Over Time (Mean Aggregated)', fontsize=16)
# plt.xticks(rotation=45)
# plt.grid(True)

# plt.savefig('./Result/sentiment_vs_sp500_plot.png')

# # Show the plot
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the sentiment dataset
# sentiment_df = pd.read_csv('./dataset/tweets_with_contextual_sentiment.csv')
# # Load the gold prices dataset
# gold_df = pd.read_csv('./dataset/gold_2024_adj_close.csv')

# # Convert the 'Date' columns to datetime format
# sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
# gold_df['Date'] = pd.to_datetime(gold_df['Date'])

# # Merge the two datasets on the 'Date' column, keeping only common dates
# merged_df = pd.merge(sentiment_df, gold_df, on='Date', how='inner')

# # Group by Date and calculate the mean (or median) of sentiment scores and Gold prices
# aggregated_df = merged_df.groupby('Date').agg({
#     'Trump_Context_Sentiment': 'mean',  # You can replace 'mean' with 'median' if needed
#     'Harris_Context_Sentiment': 'mean',
#     'Adj Close': 'mean'
# }).reset_index()

# # Create a plot with two y-axes
# fig, ax1 = plt.subplots(figsize=(14, 8))

# # Plot Trump and Harris sentiment on the first y-axis (left)
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Sentiment Score', color='blue')
# ax1.plot(aggregated_df['Date'], aggregated_df['Trump_Context_Sentiment'], label='Trump Sentiment (Mean)', color='blue', linestyle='dashed')
# ax1.plot(aggregated_df['Date'], aggregated_df['Harris_Context_Sentiment'], label='Harris Sentiment (Mean)', color='pink', linestyle='dotted')
# ax1.tick_params(axis='y', labelcolor='blue')
# ax1.legend(loc='upper left')

# # Create a second y-axis for Gold prices (right)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Gold Price (Adj Close)', color='gold')
# ax2.plot(aggregated_df['Date'], aggregated_df['Adj Close'], label='Gold Price (Mean)', color='gold')
# ax2.tick_params(axis='y', labelcolor='gold')
# ax2.legend(loc='upper right')

# # Add a title and format the x-axis
# plt.title('Trump and Harris Sentiment vs Gold Prices Over Time (Mean Aggregated)', fontsize=16)
# plt.xticks(rotation=45)
# plt.grid(True)

# # Save the plot as an image
# plt.savefig('./Result/sentiment_vs_gold_plot.png')

# # Show the plot
# plt.tight_layout()
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the sentiment dataset
sentiment_df = pd.read_csv('./dataset/tweets_with_contextual_sentiment.csv')

# Load financial datasets
sp500_df = pd.read_csv('./dataset/sp500_2024_adj_close.csv')
gold_df = pd.read_csv('./dataset/gold_2024_adj_close.csv')
treasury_df = pd.read_csv('./dataset/treasury_yield_2024_adj_close.csv')
# polymarket_df = pd.read_csv('./dataset/polymarket_daily_election.csv')
russell_df = pd.read_csv('./dataset/russell_2000_adj_close_2024.csv')

# Convert the 'Date' columns to datetime format
sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
sp500_df['Date'] = pd.to_datetime(sp500_df['Date'])
gold_df['Date'] = pd.to_datetime(gold_df['Date'])
treasury_df['Date'] = pd.to_datetime(treasury_df['Date'])
# polymarket_df['Date'] = pd.to_datetime(polymarket_df['Date (UTC)'])
russell_df['Date'] = pd.to_datetime(russell_df['Date'])

# Merge all datasets on Date column
merged_sp500 = pd.merge(sentiment_df, sp500_df[['Date', 'Adj Close']], on='Date', how='inner')
merged_gold = pd.merge(sentiment_df, gold_df[['Date', 'Adj Close']], on='Date', how='inner')
merged_treasury = pd.merge(sentiment_df, treasury_df[['Date', 'Adj Close']], on='Date', how='inner')
# merged_polymarket = pd.merge(sentiment_df, polymarket_df[['Date', 'Polymarket']], on='Date', how='inner')
merged_russell = pd.merge(sentiment_df, russell_df[['Date', 'Adj Close']], on='Date', how='inner')

# Aggregating mean values by Date
def aggregate_data(df, price_column):
    return df.groupby('Date').agg({
        'Trump_Context_Sentiment': 'mean',
        'Harris_Context_Sentiment': 'mean',
        price_column: 'mean'
    }).reset_index()

# Aggregate for each financial variable
sp500_agg = aggregate_data(merged_sp500, 'Adj Close')
gold_agg = aggregate_data(merged_gold, 'Adj Close')
treasury_agg = aggregate_data(merged_treasury, 'Adj Close')
# polymarket_agg = aggregate_data(merged_polymarket, 'Polymarket')
russell_agg = aggregate_data(merged_russell, 'Adj Close')

# Function to create a dual-axis plot
def plot_sentiment_vs_financial(df, financial_label, financial_column, financial_color, filename):
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Plot Trump and Harris sentiment on the first y-axis (left)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sentiment Score', color='blue')
    ax1.plot(df['Date'], df['Trump_Context_Sentiment'], label='Trump Sentiment (Mean)', color='blue', linestyle='dashed')
    ax1.plot(df['Date'], df['Harris_Context_Sentiment'], label='Harris Sentiment (Mean)', color='pink', linestyle='dotted')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Create a second y-axis for financial values (right)
    ax2 = ax1.twinx()
    ax2.set_ylabel(financial_label, color=financial_color)
    ax2.plot(df['Date'], df[financial_column], label=f'{financial_label} (Mean)', color=financial_color)
    ax2.tick_params(axis='y', labelcolor=financial_color)
    ax2.legend(loc='upper right')

    # Add a title and format the x-axis
    plt.title(f'Trump and Harris Sentiment vs {financial_label} Over Time (Mean Aggregated)', fontsize=16)
    plt.xticks(rotation=45)
    plt.grid(True)

    # Save the plot as an image
    plt.savefig(f'./Result/{filename}.png')

    # Show the plot
    plt.tight_layout()
    plt.show()

# Plot and save for all financial variables
plot_sentiment_vs_financial(sp500_agg, 'S&P 500', 'Adj Close', 'green', 'sentiment_vs_sp500')
plot_sentiment_vs_financial(gold_agg, 'Gold Price', 'Adj Close', 'gold', 'sentiment_vs_gold')
plot_sentiment_vs_financial(treasury_agg, 'Treasury Yield', 'Adj Close', 'orange', 'sentiment_vs_treasury')
# plot_sentiment_vs_financial(polymarket_agg, 'Polymarket', 'Polymarket', 'purple', 'sentiment_vs_polymarket')
plot_sentiment_vs_financial(russell_agg, 'Russell 2000', 'Adj Close', 'brown', 'sentiment_vs_russell')
