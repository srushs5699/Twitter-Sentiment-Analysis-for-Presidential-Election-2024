import nltk
from nltk.corpus import stopwords
from nltk import FreqDist
import string
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Step 1: Input Tweets (assuming 'Cleaned_Text' column contains preprocessed tweet text and 'Sentiment' contains labels)
tweets_df = pd.read_csv('./dataset/preprocessed_tweets.csv')

# Step 3: Assign number of features n[i]
n_values = [10, 100, 1000, 10000, 15000]

# Step 4: Define word scoring using Chi-Square Test
def wordscore_chi_square(tweets_df, n):
    # Convert text data to features (bag of words) using CountVectorizer
    vectorizer = CountVectorizer(max_features=n)
    X = vectorizer.fit_transform(tweets_df['Cleaned_Text'])
    
    # Target (Sentiment): Assuming Sentiment is binary (1 for positive, 0 for negative)
    y = tweets_df['Sentiment'].values
    
    # Chi-Square Test
    chi_scores, p_values = chi2(X, y)
    
    # Build a dictionary of word scores based on Chi-Square test
    word_scores = {}
    feature_names = vectorizer.get_feature_names_out()
    for i, score in enumerate(chi_scores):
        word_scores[feature_names[i]] = score
    
    return word_scores

# Step 5: Find the best words by sorting word scores
def find_best_words(word_scores, n):
    sorted_words = sorted(word_scores.items(), key=lambda item: item[1], reverse=True)
    best_words = [word for word, score in sorted_words[:n]]
    return best_words

# Step 6: Evaluation (Here, it's a placeholder function)
def evaluate(best_word_features, tweets_df):
    # Placeholder function for evaluation
    # This is where you'd implement a classifier using the selected best word features
    print(f"Evaluating with {len(best_word_features)} features.")

# Step 7: Main loop to process for n[i]
for n in n_values:
    print(f"\nProcessing for n = {n} features:")
    
    # Step 4: Calculate word scores using Chi-Square
    word_scores = wordscore_chi_square(tweets_df, n)
    
    # Step 5: Find the best words
    best_words = find_best_words(word_scores, n)
    
    # Step 6: Evaluate the best word features
    evaluate(best_words, tweets_df)
