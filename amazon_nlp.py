import spacy
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample Amazon reviews (replace with real dataset)
reviews = [
    "The Samsung Galaxy phone is amazing, great camera quality!",
    "I bought a Nike shoe, but it wore out quickly.",
    "Apple AirPods have excellent sound, highly recommend.",
    "The Canon camera broke after a week, very disappointing.",
    "Love my Adidas backpack, super durable!"
]

# Perform NER and sentiment analysis
for review in reviews:
    doc = nlp(review)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG"]]
    sentiment = TextBlob(review).sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    print(f"Review: {review}")
    print(f"Entities: {entities}")
    print(f"Sentiment: {sentiment_label} (Score: {sentiment:.2f})\n")