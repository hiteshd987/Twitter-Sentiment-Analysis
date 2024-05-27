import pandas as pd
import emoji

def load_dataset(path):
    twitter_data = pd.read_csv(path, header=None, encoding="ISO-8859-1", names=["target", "ids", "date", "flag", "user", "text"])
    twitter_data['text'] = twitter_data['text'].apply(emoji.demojize)

    twitter_data = twitter_data.sample(n=60000,random_state=42)
    twitter_data["target"] = twitter_data["target"].replace(4,1)
    twitter_data["target"].value_counts()

    print(twitter_data["target"].value_counts())
    
    twitter_data.dropna(inplace=True)
    return twitter_data;
