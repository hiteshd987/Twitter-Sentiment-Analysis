import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def tweets_processed(tweet):
    # Convert to lowercase
    tweet = tweet.lower()

    if tweet.startswith('rt @'):
        return None  # Return None to indicate it's a retweet

    # Remove mentions, hashtags, URLs
    tweet = re.sub(r'@[^\s]+|#[^\s]+|https?://[^\s]+', '', tweet)

    # any character that is not a word character or whitespace
    pattern = r'[^\w\s]'

    # Replace special characters with empty strings
    tweet = re.sub(pattern, '', tweet)

    # Tokenize using TweetTokenizer
    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(tweet)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

