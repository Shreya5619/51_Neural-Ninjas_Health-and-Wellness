import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "../data/bangalore_hospitals_feedback.csv"  # Adjust path as needed
feedback_data = pd.read_csv(file_path)

# Perform sentiment analysis
def analyze_sentiment(feedback):
    analysis = TextBlob(feedback)
    # Polarity ranges from -1 (negative) to 1 (positive)
    polarity = analysis.sentiment.polarity
    return polarity

# Add a new column for sentiment polarity
feedback_data['Sentiment Polarity'] = feedback_data['Feedback Text'].apply(analyze_sentiment)

# Categorize sentiment based on polarity
def categorize_sentiment(polarity):
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

feedback_data['Sentiment Category'] = feedback_data['Sentiment Polarity'].apply(categorize_sentiment)

# Save results to a new CSV file
output_file = "../results/feedback_sentiment_analysis.csv"
feedback_data.to_csv(output_file, index=False)

print(f"Sentiment analysis completed! Results saved to {output_file}")

# Visualization Section
# Set the style for the plots
sns.set(style="whitegrid")

# Countplot for Sentiment Categories
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment Category', data=feedback_data, palette='viridis')
plt.title("Sentiment Category Distribution", fontsize=16)
plt.xlabel("Sentiment Category", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()

# Distribution of Sentiment Polarity
plt.figure(figsize=(10, 6))
sns.histplot(feedback_data['Sentiment Polarity'], bins=30, kde=True, color='blue')
plt.title("Distribution of Sentiment Polarity", fontsize=16)
plt.xlabel("Sentiment Polarity", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

# Average Sentiment Rating by Hospital
plt.figure(figsize=(12, 6))
avg_sentiment = feedback_data.groupby('Hospital Name')['Sentiment Polarity'].mean().sort_values()
sns.barplot(x=avg_sentiment, y=avg_sentiment.index, palette='coolwarm')
plt.title("Average Sentiment Polarity by Hospital", fontsize=16)
plt.xlabel("Average Sentiment Polarity", fontsize=12)
plt.ylabel("Hospital Name", fontsize=12)
plt.show()
