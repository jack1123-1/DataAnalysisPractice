import pandas as pd
from food_review_sentiment import VaderSentimentEngine

df = pd.read_csv("Reviews.csv")

vader = VaderSentimentEngine()

df["vader_compound"] = df["Summary"].astype(str).apply(vader.score)
df["vader_sentiment"] = df["vader_compound"].apply(vader.label)

print(df["vader_sentiment"])
