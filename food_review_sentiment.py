from nltk.sentiment import SentimentIntensityAnalyzer

class VaderSentimentEngine:
	def __init__(self):
		self.analyzer = SentimentIntensityAnalyzer()

	def score(self, text):
		return self.analyzer.polarity_scores(text)["compound"]

	def label(self, compound_score):
		if compound_score >= 0.05:
			return "positive"
		elif compound_score <= -0.05:
			return "negative"
		else:
			return "neutral"

	def analyze(self, text):
		compound = self.score(text)
		return compound, self.label(compound)
