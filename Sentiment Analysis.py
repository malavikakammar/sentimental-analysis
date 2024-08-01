#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis on US Airline Reviews

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

import os
os.chdir('C:\\Users\\malav\\OneDrive\\Desktop')
df = pd.read_csv("Tweets.csv")


# In[2]:


df.head()


# In[23]:


df.columns


# In[4]:


tweet_df = df[['text','sentiment','textID']]
print(tweet_df.shape)
tweet_df.head(5)


# In[22]:


tweet_df = tweet_df[tweet_df['sentiment'] != 'neutral']
print(tweet_df.shape)
tweet_df.head(5)


# In[21]:


tweet_df["sentiment"].value_counts()


# In[6]:


sentiment_label = tweet_df.sentiment.factorize()
sentiment_label


# In[7]:preparing a dataset for training


tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)


# In[8]:


print(tokenizer.word_index)


# In[9]:vectorisation


print(tweet[0])
print(encoded_docs[0])


# In[10]:


print(padded_sequence[0])


# In[11]:creating a model using multiple layers


embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  
print(model.summary()) 


# In[12]:loss function and optimization


history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)


# In[16]:


plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()
plt.savefig("Accuracy plot.jpg")


# In[25]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.savefig("Loss plot.jpg")


# In[18]:


def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])


# In[19]:


test_sentence1 = "WOW, i AM REALLY MiSSiN THE FAM(iLY) TODAY"
predict_sentiment(test_sentence1)

test_sentence2 = "happy 1 year! <3."
predict_sentiment(test_sentence2)


# In[ ]:
neg = df[df['sentiment']=='negative']
pos = df[df['sentiment']=='positive']

import plotly.graph_objs as go
fig = go.Figure()
for col in pos.columns:
    fig.add_trace(go.Scatter(x=pos['sentiment'], y=pos['textID'],
                             name = col,
                             mode = 'markers+lines',
                             line=dict(shape='linear'),
                             connectgaps=True,
                             line_color='green'
                             )
                 )
for col in neg.columns:
    fig.add_trace(go.Scatter(x=neg['sentiment'], y=neg['textID'],
                             name = col,
                             mode = 'markers+lines',
                             line=dict(shape='linear'),
                             connectgaps=True,
                             line_color='red'
                             )
       
                 )
fig.show()





# %%
import plotly.express as px
import os
import pandas as pd
os.chdir('C:\\Users\\malav\\OneDrive\\Desktop')
df = pd.read_csv("Tweets.csv")
df.head()

fig = px.scatter(df, x="sentiment", y="textID", 
                 width=800, height=400)

fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="white",
)

fig.show()
# %%
