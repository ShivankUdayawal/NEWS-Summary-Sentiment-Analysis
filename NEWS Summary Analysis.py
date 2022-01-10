#!/usr/bin/env python
# coding: utf-8

# # NEWS Summary
# 
# **Generating short length descriptions of news articles.**
# 
# <img src="01.jpg" width=800 height=60 />

# ## Text summarization using machine learning techniques
# 
# ### A sequence-to-sequence model using an Encoder-Decoder with Attention
# 
# The encoder-decoder model for recurrent neural networks is an architecture for sequence-to-sequence prediction problems. It comprised two parts:
# 
# **1. Encoder:** The encoder is responsible for stepping through the input time steps, read the input words one by one and encoding the entire sequence into a fixed length vector called a context vector.
# 
# **2. Decoder:** The decoder is responsible for stepping through the output time steps while reading from the context vector, extracting the words one by one. The trouble with seq2seq is that the only information that the decoder receives from the encoder is the last encoder hidden state which is like a numerical summary of an input sequence. So, for a long input text, we expect the decoder to use just this one vector representation to output a translation. This might lead to catastrophic forgetting.
# 
# 
# To solve this problem, the attention mechanism was developed. 
# 
# ***Attention*** is proposed as a method to both align and translate. It identifies which parts of the input sequence are relevant to each word in the output (alignment) and use that relevant information to select the right output (translation). So instead of encoding the input sequence into a single fixed context vector (reason for the mentioned bad performance), the attention model develops a context vector that is filtered specifically for each output time step. Attention provides the decoder with information from every encoder hidden state. With this setting, the model can selectively focus on useful parts of the input sequence and hence, learn the alignment between them.
# 
# In the next few sections we will go through the whole process: Load the datasets and vector representation, build the vocabulary, define the encoder, decoder and attention mechanism. Then we will code the train stage, iterating over the datasets, and finally we will make the predictions for the validation dataset to get the value of the metrics of interest.
# 
# 
# ## Content :
# 
# The dataset consists of 4515 examples and contains Author_name, Headlines, Url of Article, Short text, Complete Article. I gathered the summarized news from Inshorts and only scraped the news articles from Hindu, Indian times and Guardian. Time period ranges from febrauary to august 2017.
# 
# 
# ## Data Source :
# 
# https://www.kaggle.com/sunnysai12345/news-summary

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import unicodedata
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Layer
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords 
from textblob import TextBlob
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import warnings
warnings.filterwarnings('ignore')


# ## Importing the Dataset

# In[2]:


data = pd.read_csv("news_summary.csv", encoding = 'unicode_escape')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data['author'].unique()


# In[7]:


sns.set_style("whitegrid")
plt.figure(figsize = (18, 6))
sns.countplot(data['author'])
plt.xticks(rotation = 90);
plt.xlabel("--------------- Author ------------", fontsize = 18);


# * **Chhavi Tyagi** wrote highest number of news headlines which are more than 500
# * **Tarun Khanna** wrote second highest number of news headlines which are more than 300
# 
# * **Rini Sinha**, **Daisy Mowke**, **Aarushi Maheshwari**, **Mansha Mahajan**, **Saloni Tandon**, **Dishant Sharma**, **Prashanti Moktan**, **Sumedha Sehra**, **Shubhodeep Datta**, **Abhishek Bansal**, **Deepali Aggarwal**, **Arshiya Chopra**, **Sonu Kumari**, **Parmeet Kaur** these are the authors who wrote more than 100 news headlines
# 
# * **Jatan Desai**, **Diksha Dhiman**, **Sultan Mirza**, **Trivedi Bhutnath** these are new authors who wrote only 1 headline.

# In[8]:


data['date'] = pd.to_datetime(data['date'])


# In[9]:


data["day"]      = pd.to_datetime(data['date']).dt.day
data["month"]    = pd.to_datetime(data['date']).dt.month
data["year"]     = pd.to_datetime(data['date']).dt.year
data["weekday"]  = pd.to_datetime(data['date']).dt.weekday


# In[10]:


data['weekday'] = data['weekday'].replace({0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 
                                           6:'Sunday'})


# In[11]:


data['month'] = data['month'].replace({1:'January', 2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 
                                       8:'August', 9:'September', 10:'October', 11:'November', 12:'December'})


# In[12]:


plt.figure(figsize = (18, 6))
sns.countplot(data['day'])
plt.xticks(rotation = 0);
plt.xlabel("--------------- Day ------------", fontsize = 18);


# In[13]:


plt.figure(figsize = (18, 6))
sns.countplot(data['month'])
plt.xticks(rotation = 0);
plt.xlabel("--------------- Month ------------", fontsize = 18);


# * Maximum news headlines are in **July** month which is more than 1400
# * **January**, **March**, **April**, **June** are the months in which morethan 400 news headlines published
# * **August** is the month in which least news headline published

# In[14]:


plt.figure(figsize = (15,5))
sns.countplot(data['year'])
plt.xticks(rotation = 0);
plt.xlabel("--------------- Year ------------", fontsize = 18);


# * Only **8.2%** news headlines are from **2016**
# * **91.8%** news headlines are from **2017**

# In[15]:


plt.figure(figsize = (18, 6))
sns.countplot(data['weekday'])
plt.xticks(rotation = 0);
plt.xlabel("--------------- Weekday ------------", fontsize = 18);


# * Max no of news headlines are in **Wednesday** which is more than 700
# * **Tuesday**, **Thursday**, **Friday** have more than 600 news headline
# * **Weekend** have least no of news headlines

# In[16]:


data.head()


# In[17]:


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(background_color = 'white', max_words = 4500, max_font_size = 40, scale = 1, 
                          random_state = 42).generate(str(data))

    fig = plt.figure(1, figsize = (20,6))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(data['headlines'])


# In[18]:


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(background_color = 'white', max_words = 4500, max_font_size = 40, scale = 1, 
                          random_state = 42).generate(str(data))

    fig = plt.figure(1, figsize = (20,6))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(data['read_more'])


# In[19]:


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(background_color = 'white', max_words = 4500, max_font_size = 40, scale = 1, 
                          random_state = 42).generate(str(data))

    fig = plt.figure(1, figsize = (20,6))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(data['text'])


# In[20]:


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(background_color = 'white', max_words = 4500, max_font_size = 40, scale = 1, 
                          random_state = 42).generate(str(data))

    fig = plt.figure(1, figsize = (20,6))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(data['ctext'])


# In[21]:


data.drop(['date', 'read_more', 'day', 'month', 'year', 'weekday'], axis = 1, inplace = True)


# In[22]:


data.head()


# In[23]:


data['ctext'][0]


# In[24]:


data['ctext'] = data['ctext'].astype("str").astype("string")


# ## PreProcessing of Text
# ### Text Normalization
# ### Removing Html Strips & Noise Text
# ### Removing Special Characters

# In[25]:


data['cleaned_text'] = data['ctext'].replace(r'\'|\"|\,|\.|\?|\+|\-|\/|\=|\(|\)|\n|"', '', regex = True)

# Replacing few double spaces with single space
data['cleaned_text'] = data['ctext'].replace("  ", " ")

# remove emoticons form the tweets
data['cleaned_text'] = data['ctext'].replace(r'<ed>', '', regex = True)
data['cleaned_text'] = data['ctext'].replace(r'\B<U+.*>|<U+.*>\B|<U+.*>', '', regex = True)

# convert tweets to lowercase
data['cleaned_text'] = data['ctext'].str.lower()
    
#remove user mentions
data['cleaned_text'] = data['ctext'].replace(r'^(@\w+)', "", regex = True)
    
#remove 'rt' in the beginning
data['cleaned_text'] = data['ctext'].replace(r'^(rt @)', "", regex = True)
    
#remove_symbols
data['cleaned_text'] = data['ctext'].replace(r'[^a-zA-Z0-9]', " ", regex = True)

#remove punctuations 
data['cleaned_text'] = data['ctext'].replace(r'[[]!"#$%\'()\*+,-./:;<=>?^_`{|}]+', "", regex = True)

#remove_URL(x):
data['cleaned_text'] = data['ctext'].replace(r'https.*$', "", regex = True)

#remove 'amp' in the text
data['cleaned_text'] = data['ctext'].replace(r'amp', "", regex = True)

#remove words of length 1 or 2 
data['cleaned_text'] = data['ctext'].replace(r'\b[a-zA-Z]{1,2}\b', '', regex = True)

#remove extra spaces in the tweet
data['cleaned_text'] = data['ctext'].replace(r'^\s+|\s+$', " ", regex = True)


# In[26]:


data['cleaned_text'] = data['cleaned_text'].astype("str").astype("string")


# ### Removing Html Strips & Noise Text

# In[27]:


#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

#Apply function on review column
data['cleaned_text'] = data['ctext'].apply(denoise_text)


# ### Removing Special Characters

# In[28]:


def remove_special_characters(text, remove_digits = True):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern,'',text)
    return text

#Apply function on review column
data['cleaned_text'] = data['ctext'].apply(remove_special_characters)


# ## Tokenization

# In[29]:


tokenizer = ToktokTokenizer()

stopword_list = nltk.corpus.stopwords.words('english')


# ### Text Stemming

# In[30]:


def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

data['cleaned_text'] = data['cleaned_text'].apply(simple_stemmer)


# ### Removing Stopwords

# In[31]:


stop = set(stopwords.words('english'))

def remove_stopwords(text, is_lower_case = False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


data['cleaned_text'] = data['cleaned_text'].apply(remove_stopwords)


# ### Polarity

# In[32]:


data['sentiment'] = data['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
data.head()


# In[33]:


def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
data['sentiments'] = data['sentiment'].apply(getAnalysis )


# In[34]:


data.head()


# In[35]:


data['sentiments'].value_counts()


# In[36]:


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(background_color = 'white', max_words = 4500, max_font_size = 40, scale = 1, 
                          random_state = 42).generate(str(data))

    fig = plt.figure(1, figsize = (20,10))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(wordcloud)
    plt.show()
    
# print wordcloud
show_wordcloud(data['cleaned_text'])


# In[37]:


from PIL import Image


# In[38]:


stopwords = set(STOPWORDS)
mask = np.array(Image.open(r"D:\Projects\News Summary\04.JPG"))
wordcloud = WordCloud(width = 3000, height = 2000, random_state = 1, background_color = 'white', colormap = 'Set2', 
                      collocations = False, mode = "RGBA", max_words = 4000, 
                      mask = mask).generate(' '.join(data['cleaned_text']))

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize = [20,20])
plt.imshow(wordcloud.recolor(color_func = image_colors), interpolation = "bilinear")
plt.axis("off") 
plt.show()


# https://www.analyticsvidhya.com/blog/2021/08/creating-customized-word-cloud-in-python/

# In[39]:


mask = np.array(Image.open(r"D:\Projects\News Summary\05.JPG"))
wordcloud = WordCloud(width = 3000, height = 2000, random_state = 1, background_color = 'white', colormap = 'Set2', 
                      collocations = False, mode = "RGBA", max_words = 4000, 
                      mask = mask).generate(' '.join(data['cleaned_text']))

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize = [20,20])
plt.imshow(wordcloud.recolor(color_func = image_colors), interpolation = "bilinear")
plt.axis("off") 
plt.show()


# In[40]:


mask = np.array(Image.open(r"D:\Projects\News Summary\08.JPG"))
wordcloud = WordCloud(width = 3000, height = 2000, random_state = 1, background_color = 'white', colormap = 'Set2', 
                      collocations = False, mode = "RGBA", max_words = 4000, 
                      mask = mask).generate(' '.join(data['cleaned_text']))

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize = [20,20])
plt.imshow(wordcloud.recolor(color_func = image_colors), interpolation = "bilinear")
plt.axis("off") 
plt.show()


# In[41]:


mask = np.array(Image.open(r"D:\Projects\News Summary\07.JPG"))
wordcloud = WordCloud(width = 3000, height = 2000, random_state = 1, background_color = 'white', colormap = 'Set2', 
                      collocations = False, mode = "RGBA", max_words = 4000, 
                      mask = mask).generate(' '.join(data['cleaned_text']))

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize = [20,20])
plt.imshow(wordcloud.recolor(color_func = image_colors), interpolation = "bilinear")
plt.axis("off") 
plt.show()


# In[42]:


mask = np.array(Image.open(r"D:\Projects\News Summary\06.JPG"))
wordcloud = WordCloud(width = 3000, height = 2000, random_state = 1, background_color = 'white', colormap = 'Set2', 
                      collocations = False, mode = "RGBA", max_words = 4000, 
                      mask = mask).generate(' '.join(data['cleaned_text']))

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize = [20,20])
plt.imshow(wordcloud.recolor(color_func = image_colors), interpolation = "bilinear")
plt.axis("off") 
plt.show()


# In[43]:


sns.set_style("whitegrid")
plt.figure(figsize = (18,5))
sns.countplot(data['author'], hue = data['sentiments'])
plt.xticks(rotation = 90);
plt.xlabel("--------------- Author ------------", fontsize = 18);


# * most of the headlines form the author show **Positive Sentiments** 
# * **Sonu Kumari**, **Parmeet Kaur**, **Radhika Chugh**, **Krishna Veera Vanamali**, **Rini Kapoor** & **Parichit Maira** these authors have almost equal ratio of **Positive - Negative Sentiments**

# In[44]:


data.head()


# ### TF-IDF usage

# In[45]:


corpus = data.cleaned_text  ## Collection of documents 
vectorizer = TfidfVectorizer(stop_words = 'english', analyzer = 'word')
print(vectorizer)

X = vectorizer.fit_transform(corpus)
print(X[:5]) 


# In[46]:


idf = vectorizer.idf_
print(idf)


# In[47]:


vectorizer.vocabulary_


# In[48]:


vectorizer.get_feature_names()


# In[49]:


col = ['feat_'+ i for i in vectorizer.get_feature_names()]
print(col[1:5])
print(X[1:5])


# In[ ]:




