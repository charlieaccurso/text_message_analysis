import pandas as pd
import random
from gensim.parsing.preprocessing import preprocess_string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# load dataframe
df = pd.read_csv('clean_nus_sms.csv')

# only analyze the first n messages (due to computational limits)
df= df.head(5000)

# ensure all values are strings
df['Message'] = df['Message'].astype(str)

# preprocess text messages using gensim's preprocessing function
def preprocess_text(text):
    return preprocess_string(text)

df['processed_message'] = df['Message'].apply(preprocess_text)

# create tagged documents for each message
tagged_docs = [TaggedDocument(words=d, tags=[str(i)]) for i, d in enumerate(df['processed_message'])]
# A TaggedDocument object represents a document with a unique identifier (tag) and 
# its corresponding text. This list of TaggedDocument's is then used to train the Doc2Vec model

# train the Doc2Vec model
model = Doc2Vec(tagged_docs, vector_size=50, window=10, min_count=5, workers=4, epochs=100)

# classify each message using the model
def classify_sentiment(text):
    vec = model.infer_vector(preprocess_text(text))
    inferred_docvec = model.dv.most_similar([vec])[0]
    if inferred_docvec[1] >= 0.9:
        return 'positive'
    else:
        return 'negative'

df['sentiment'] = df['Message'].apply(classify_sentiment)

# print sentiment classification for 20 random messages in the dataset
random_messages = random.sample(list(df['Message']), 20)
for message in random_messages:
    sentiment = classify_sentiment(message)
    print(f"Message: {message}\nSentiment: {sentiment}\n")