import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('vader_lexicon')

# Create sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Take input from user
text = input("Enter a sentence: ")

# Analyze sentiment
score = sia.polarity_scores(text)

print("\nSentiment Scores:")
print(score)

# Determine sentiment
if score['compound'] >= 0.05:
    print("Sentiment: POSITIVE")
elif score['compound'] <= -0.05:
    print("Sentiment: NEGATIVE")
else:
    print("Sentiment: NEUTRAL")