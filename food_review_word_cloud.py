import pandas as pd
import matplotlib.pyplot as plt
from preprocessor import unigram_preprocessor, bigram_preprocessor, trigram_preprocessor
from wordcloud import WordCloud

df = pd.read_csv("Reviews.csv")

df["trigram"] = df["Summary"].astype(str).apply(trigram_preprocessor)

positive_trigram = " ".join(df[df["Score"] == 5]["trigram"])
negative_trigram = " ".join(df[df["Score"] == 1]["trigram"])

wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=150,
        collocations=False
    ).generate(positive_trigram)

plt.figure(figsize=(12,5))
plt.imshow(wc)
plt.axis("off")
plt.title("Trigram 5 star Reviews")
plt.show()

wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=150,
        collocations=False
    ).generate(negative_trigram)

plt.figure(figsize=(12,5))
plt.imshow(wc)
plt.axis("off")
plt.title("Trigram 1 star Reviews")
plt.show()
