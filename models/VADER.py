from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def check_vader_polarity(text):
    polarity_score = analyzer.polarity_scores(text)
    if polarity_score['compound'] >= 0.05:
        return 1
    elif polarity_score['compound'] <=0.05:
        return 0
    else:
        return 2
