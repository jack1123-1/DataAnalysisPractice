import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def text_preprocessor(text):
	text = text.lower()
	text = re.sub(r"[^a-z\s]", "", text)
	tokens = word_tokenize(text)
	tokens = [word for word in tokens if word not in stop_words or word in {"not", "no", "never"}]
	tokens = [lemmatizer.lemmatize(word) for word in tokens]

	return tokens

def unigram_preprocessor(text):
	tokens = text_preprocessor(text)
	return " ".join(tokens)

def bigram_preprocessor(text):
	tokens = text_preprocessor(text)
	bigrams = ngrams(tokens, 2)
	return " ".join(["_".join(bigram) for bigram in bigrams])

def trigram_preprocessor(text):
	tokens = text_preprocessor(text)
	trigrams = ngrams(tokens, 3)
	return " ".join(["_".join(trigram) for trigram in trigrams])