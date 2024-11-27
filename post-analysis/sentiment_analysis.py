from flask import Flask, render_template
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the dataset
file_path = "../data/bangalore_hospitals_feedback.csv"  # Adjust path as needed
feedback_data = pd.read_csv(file_path)

# Perform sentiment analysis
def analyze_sentiment(feedback):
    analysis = TextBlob(feedback)
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

# Initialize Flask app
app = Flask(__name__)

# Route for the main page
@app.route('/')
def index():
    # Countplot for Sentiment Categories
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sentiment Category', data=feedback_data, palette='viridis')
    plt.title("Sentiment Category Distribution", fontsize=16)
    plt.xlabel("Sentiment Category", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    countplot_path = "static/sentiment_category_distribution.png"
    plt.savefig(countplot_path)
    plt.close()

    # Distribution of Sentiment Polarity
    plt.figure(figsize=(10, 6))
    sns.histplot(feedback_data['Sentiment Polarity'], bins=30, kde=True, color='blue')
    plt.title("Distribution of Sentiment Polarity", fontsize=16)
    plt.xlabel("Sentiment Polarity", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    polarity_dist_path = "static/sentiment_polarity_distribution.png"
    plt.savefig(polarity_dist_path)
    plt.close()

    # Average Sentiment Rating by Hospital
    plt.figure(figsize=(12, 6))
    avg_sentiment = feedback_data.groupby('Hospital Name')['Sentiment Polarity'].mean().sort_values()
    sns.barplot(x=avg_sentiment, y=avg_sentiment.index, palette='coolwarm')
    plt.title("Average Sentiment Polarity by Hospital", fontsize=16)
    plt.xlabel("Average Sentiment Polarity", fontsize=12)
    plt.ylabel("Hospital Name", fontsize=12)
    avg_sentiment_path = "static/average_sentiment_by_hospital.png"
    plt.savefig(avg_sentiment_path)
    plt.close()

    return render_template(
        'index.html',
        countplot_path=countplot_path,
        polarity_dist_path=polarity_dist_path,
        avg_sentiment_path=avg_sentiment_path,
        feedback_data=feedback_data.to_html(classes='table table-striped', index=False)
    )

# Run the Flask app
if __name__ == "__main__":
    # Create static directory if not exists
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
