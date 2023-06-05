#!/usr/bin/env python
# coding: utf-8

# **<h1 text align = center>1. Quora Question pair Similarity Case Study </h1>**

# *1* Linking colab with Gdrive.
# 
# *2* Importing Bunch of libraries.
# 
# *3* Gethering data information.

# In[10]:


from google.colab import drive
drive.mount("/content/gdrive")


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")
import re
from nltk.stem import PorterStemmer
from wordcloud import WordCloud , STOPWORDS 
import nltk
nltk.download("stopwords")
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
get_ipython().system(' pip install distance')
import distance
get_ipython().system(' pip install fuzzywuzzy')
from fuzzywuzzy import fuzz
from os import path
get_ipython().system(' pip install wordcloud')
from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
get_ipython().system(' pip install spacy')
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.metrics import accuracy_score , log_loss
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier


# <h2>1.1 Reading the data and analysing it.

# In[ ]:


df = pd.read_csv("/content/gdrive/My Drive/Quora/train.csv")
df.head()


# In[ ]:


df.describe


# In[ ]:


df.describe()


# In[ ]:


df.info()


# In[ ]:


df.columns


# <h2>1.2 Plotting the duplicate(1) and non duplicate(0) questions id.

# In[ ]:


df.groupby("is_duplicate")["id"].count().plot.bar()


# In[ ]:


print(" {}% of data points is non-duplicate (not Similar).".format(100 - round(df['is_duplicate'].mean() *100,2)))


# In[ ]:


print("{}% of data points is duplicate (Similar)." .format(round(df["is_duplicate"].mean() *100 , 2)))


# In[ ]:


# the code aims to analyze the values in the 'qid1' and 'qid2' columns of the DataFrame and provide insights about the uniqueness
# and frequency of the values present in these columns.
qids = pd.Series(df['qid1'].tolist() + df['qid2'].tolist())
unq_qstn = len(np.unique(qids))
qstn_more_than_one = np.sum(qids.value_counts() > 1)
print(qids.value_counts())


# In[ ]:


print("Number of unique question that appear more than once are {} {}%".format(qstn_more_than_one , qstn_more_than_one/unq_qstn*100))


# In[ ]:


np.where(qids==2559)
# Question id 2559 that repeated max 157 times.


# In[ ]:


print(df.loc[38200])


# Question with **qid1= *2559*** repeated max **157** times and the question is " **What are the best way to lose weight?** "

# <h2>1.3 Plotting Unique and Repeated Questions

# In[ ]:


x = ['Unique Question' , 'Repeated Question']
y = [unq_qstn , qstn_more_than_one]
plt.figure(figsize=(5,7))
sns.barplot(x= x,y=y)
plt.title("Plot showing Unique and Repeated Questions")
plt.show()


# *--> Checking the duplicate Pair of Questions*

# In[ ]:


pair_duplicates = df[["qid1","qid2","is_duplicate"]].groupby(["qid1" ,"qid2" ]).count().reset_index()
print("Number of Duplicate Questions",(pair_duplicates).shape[0] - df.shape[0])


# <h2>1.4 Plotting number of occurance of questions

# In[ ]:


plt.figure(figsize=(15,5))
plt.hist(qids.value_counts() , bins=160)
plt.yscale('log')
plt.xlabel("Number of Occurance of Questions")
plt.ylabel("Number of Questions")
plt.title("Histogram showing Number of occurance of Questions")
plt.show()
print("Maximum number of time a single question occure is {}".format(max(qids.value_counts())))


# *--> Checking the Null Values*

# In[ ]:


narow = df[df.isnull().any(1)]
narow


# OBSERVATION: Column question2 has **Two** null value and column question1 has **One** null value.

# In[ ]:


# removing the null values.
df = df.fillna('')
na_row = df[df.isnull().any(1)]
print(na_row)


# <h2>1.5 Lets try some basic **Feature Extraction** before cleaning the data

# - ____freq_qid1____
# - ____freq_qid2____
# - ____q1_len____
# - ____q2_len____
# - ____q1_n_words____
# - ____q2_n_words____
# - ____word_common____
# - ____word_total____
# - ____word_share____
# - ____freq_qid1 + freq_qid2____
# - ____freq_qid1 - freq_qid2____

# <h3> 1.5.1 The below code snippet is essentially extracting various features from the 'df' DataFrame and saving the modified DataFrame to the CSV file for further analysis or usage.

# In[ ]:


if os.path.isfile("/content/gdrive/My Drive/Quora/basic_feature_extraction.csv"): # if file.csv in my gdrive then open else create new file.csv
  df = pd.read_csv("/content/gdrive/My Drive/Quora/basic_feature_extraction.csv" , encoding="latin-1")
else:
  df["freq_qid1"] = df.groupby('qid1')['qid1'].transform('count') # checking repeating frequency of question1
  df["freq_qid2"] = df.groupby('qid2')['qid2'].transform('count') # ---do----
  df["q1_len"]    = df['question1'].str.len()   # creating length of string eg:- shubham balgotra ----o/p= 16(including space as character)
  df["q2_len"]    = df['question2'].str.len()   #----do---
  df["q1_n_words"]= df['question1'].apply(lambda row: len(row.split(" ")))   # apply() use to apply function, lambda keyword is used to define an anonymous function in Python. 
  df["q2_n_words"]= df['question2'].apply(lambda row: len(row.split(" ")))   #----do----
  
  def normalized_word_common(row):
    w1 = set(map(lambda word: word.lower().strip() , row['question1'].split(" ")))   # set() arrange in ascending order, map() allows you to process and transform all the items in an iterable without using an explicit for loop
    w2 = set(map(lambda word: word.lower().strip() , row['question2'].split(" ")))   #strip() remove whitespaces.
    return 1.0 * len(w1 & w2)   # question words split with space, perform lower() and strip() function on it and store to w1 and w2 respect. and then using AND on w1 and w2.     
  df["word_common"] = df.apply(normalized_word_common , axis=1)

  def normalized_word_total(row):
    w1 = set(map(lambda word: word.lower().strip() , row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip() , row['question2'].split(" ")))
    return 1.0 * (len(w1) + len(w2)) # question words split with space, perform lower() and strip() function on it and store to w1 and w2 respect. and then using OR on w1 and w2.
  df["word_total"] = df.apply(normalized_word_total , axis=1)

  def normalized_word_share(row):
    w1 = set(map(lambda word: word.lower().strip() , row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip() , row['question2'].split(" ")))
    return 1.0 * len(w1 & w2)/(len(w1)+len(w2)) # question words split with space, perform lower() and strip() function on it and store to w1 and w2 respect. and then using AND/OR on w1 and w2.
  df["word_share"] = df.apply(normalized_word_share , axis=1)

  df["freq_qid1+qid2"] = df["freq_qid1"] + df["freq_qid2"]   # Adding occurance of both questions 
  df["freq_qid1-qid2"] = abs(df["freq_qid1"] - df["freq_qid2"]) # Subtracting occurance of questions

  df.to_csv("/content/gdrive/My Drive/Quora/basic_feature_extraction.csv", index=False)

df.head(2)


# In[ ]:


df.shape


# In[ ]:


print("Minimum length of words in question1 {}".format(min(df['q1_n_words'])))
print("Minimum length of words in question2 {}\n".format(min(df['q2_n_words'])))

print("Maximum length of words in question1 {}".format(max(df['q1_n_words'])))
print("Maximum length of words in question2 {}\n".format(max(df['q2_n_words'])))
print("Number of questions in question1 containing ONE word only = {}".format(len(df[df['q1_n_words']==1])))
print("Number of questions in question2 containing ONE word only = {}".format(len(df[df['q2_n_words']==1])))


# <h3> 1.5.2 Analysing feature Word_share.

# In[ ]:


plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
plt.title("Question sharing common words")
sns.violinplot(x="is_duplicate", y="word_share", data=df[0:])

plt.subplot(1,2,2)
sns.distplot(df[df["is_duplicate"] == 0.0]["word_share"][0:], label='0', color='red')
sns.distplot(df[df['is_duplicate'] == 1.0]["word_share"][0:], label='1', color='blue')
plt.legend()
plt.title("Distribution Plot for words share")
plt.show()


# <h3> 1.5.3 Analysing feature Word_common.
# 
# ---
# 
# 

# In[ ]:


plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
plt.title("Questions having common words")
sns.violinplot(x='is_duplicate', y='word_common', data=df)

plt.subplot(1,2,2)
plt.title("Distribution Plot for common words")
sns.distplot(df[df['is_duplicate'] == 0.0]['word_common'][0:], label='0', color='red')
sns.distplot(df[df['is_duplicate'] == 1.0]['word_common'][0:], label='1', color='blue')
plt.legend()
plt.show()


# <h2>1.6 Preprocessing Text</h1>

# <h3>1.6.1 The preprocess() function applies various text cleaning and normalization techniques to the input text, such as converting to lowercase, replacing specific patterns or words, stemming, and removing HTML tags. These steps help standardize and clean the text data for further analysis or processing.

# In[ ]:


STOP_WORDS = stopwords.words("english")
SAFE_DIV = 0.00001

def preprocess(x):
  x = str(x).lower()
  x = x.replace(',000,000', 'm').replace(',000', 'k').replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
  x = re.sub(r"([0-9]+)000000", r"\1m",x)
  x = re.sub(r"([0-9]+)000" , r"\1k",x)

  porter = PorterStemmer()
  pattern = re.compile("\W")

  if type(x) == type(''):
      x = re.sub(pattern, " ",x)

  if type(x) ==  type(''):
      x = porter.stem(x)
      example = BeautifulSoup(x)
      x = example.get_text()

  return x


# <h3>1.6.2 The token_feature list, containing calculated features, is returned by the function. These features can be useful for further analysis or modeling tasks involving natural language processing and question similarity. Also creating the new datafield with these new features.

# In[ ]:


def get_token_feature(q1,q2):
  token_feature = [0.0]*10
  q1_token = q1.split()
  q2_token = q2.split()
  if(len(q1_token) == 0 or len(q2_token) == 0):
      return token_feature
  q1_stop = set([word for word in q1_token if word in STOPWORDS])
  q2_stop = set([word for word in q2_token if word in STOPWORDS])

  q1_word = set([word for word in q1_token if word not in STOPWORDS])
  q2_word = set([word for word in q2_token if word not in STOPWORDS])

  common_word_count  = len(q1_word.intersection(q2_word))
  common_stop_count  = len(q2_stop.intersection(q2_stop))
  common_token_count = len(set(q1_token).intersection(set(q2_token)))

  token_feature[0] = common_word_count / (min(len(q1_word) , len(q2_word)) + SAFE_DIV)
  token_feature[1] = common_word_count / (max(len(q1_word) , len(q2_word)) + SAFE_DIV)
  token_feature[2] = common_stop_count / (min(len(q1_stop) , len(q2_word)) + SAFE_DIV)
  token_feature[3] = common_stop_count / (max(len(q1_stop) , len(q2_stop)) + SAFE_DIV)
  token_feature[4] = common_token_count / (min(len(q1_token) , len(q2_token)) + SAFE_DIV)
  token_feature[5] = common_token_count / (max(len(q1_token) , len(q2_token)) + SAFE_DIV)
  token_feature[6] = int(q1_token[-1] == q2_token[-1])
  token_feature[7] = int(q1_token[0] == q2_token[0])
  token_feature[8] = abs(len(q1_token) - len(q2_token))
  token_feature[9] = (len(q1_token) + len(q2_token)) / 2
  return token_feature

# Getting longest common sub string in question1 and question2
def get_longest_common_substring(a,b):
  string = list(distance.lcsubstrings(a,b))
  if len(string) == 0:
    return 0
  else:
    return len(string[0]) / (min(len(a) , len(b)) +1)

def extract_features(df):

  df['question1'] = df['question1'].fillna('').apply(preprocess)
  df['question2'] = df['question2'].fillna('').apply(preprocess)

  token_feature  = df.apply(lambda x: get_token_feature (x['question1'], x['question2']), axis = 1)
  
  df['cwc_min']                     = list(map(lambda x: x[0] , token_feature))
  df['cwc_max']                     = list(map(lambda x: x[1] , token_feature))
  df['csc_min']                     = list(map(lambda x: x[2] , token_feature))
  df['csc_max']                     = list(map(lambda x: x[3] , token_feature))
  df['ctc_min']                     = list(map(lambda x: x[4] , token_feature))
  df['ctc_max']                     = list(map(lambda x: x[5] , token_feature))
  df['last_word_common']            = list(map(lambda x: x[6] , token_feature))
  df['first_word_common']           = list(map(lambda x: x[7] , token_feature))
  df['abs_len_diff']                = list(map(lambda x: x[8] , token_feature))
  df['mean_ratio']                  = list(map(lambda x: x[9] , token_feature))

  df['fuzz_ratio']                  = df.apply(lambda x: fuzz.QRatio(x['question1'] , x['question2']), axis=1)
  df['fuzz_partial_ratio']          = df.apply(lambda x: fuzz.partial_ratio(x['question1'] , x['question2']), axis=1)
  df['token_set_ratio']             = df.apply(lambda x: fuzz.token_set_ratio(x['question1'] , x['question2']), axis=1)
  df['token_sort_ratio']            = df.apply(lambda x: fuzz.token_sort_ratio(x['question1'] , x['question2']), axis=1)
  df['longest_common_substring']    = df.apply(lambda x: get_longest_common_substring(x['question1'] , x['question2']), axis=1)
  return df


# In[ ]:


if os.path.isfile("/content/gdrive/My Drive/Quora/nlp_feature_train.csv"):
  df2 = pd.read_csv("/content/gdrive/My Drive/Quora/nlp_feature_train.csv")
else:
  df2 = pd.read_csv("/content/gdrive/My Drive/Quora/train.csv")
  df2 = extract_features(df2)
  df2.to_csv("/content/gdrive/My Drive/Quora/nlp_feature_train.csv" , index=False)


# In[ ]:


df2.head(2)


# <h2>1.7 Analysing Extracting Features</h2>

# In[ ]:


# Creating and Saving positive and negative questios.
df_duplicate    = df2[df2['is_duplicate'] == 1]
df_nonduplicate = df2[df2['is_duplicate'] == 0]

p = np.dstack([df_duplicate["question1"] , df_duplicate["question2"]]).flatten()
n = np.dstack([df_nonduplicate["question1"] , df_nonduplicate["question2"]]).flatten()

print("Number of data points in duplicate questions are {}".format(len(p)))
print("Number of data points in non_duplicate questions are {}".format(len(n)))

np.savetxt("/content/gdrive/My Drive/Quora/train_p.txt", p , fmt='%s', delimiter=' ')
np.savetxt("/content/gdrive/My Drive/Quora/train_n.txt", n , fmt='%s', delimiter=' ')


# In[ ]:


link = path.dirname("/content/gdrive/My Drive/Quora/")
textp_w = open(path.join(link, "train_p.txt")).read()
textn_w = open(path.join(link, "train_n.txt")).read()


# In[ ]:


stopwords = set(STOPWORDS)
stopwords.add(" ")
stopwords.remove("not")
stopwords.remove("no")
print("Number of words in duplicate pair of questions :",format(len(textp_w)))
print("Number of words in non duplicate pair of questions :",format(len(textn_w)))


# <h2>1.8 Building Wordloud

# In[ ]:


wc = WordCloud(stopwords=stopwords , background_color='white' , max_words=len(textp_w))
wc.generate(textp_w)
plt.title("This is for duplicate pairs")
plt.imshow(wc , interpolation='bilinear',)
#plt.axis('off')
plt.show()


# In[ ]:


wc = WordCloud(stopwords=stopwords , background_color='white' , max_words=len(textn_w))
wc.generate(textn_w)
plt.title("This is for non-duplicate pairs")
plt.imshow(wc , interpolation='bilinear')
#plt.axis("off")
plt.show()


# In[ ]:


df2.shape[0]
sns.pairplot(df2 , hue='is_duplicate' , vars = ['cwc_min' , 'csc_min', 'ctc_min' , 'token_sort_ratio'])
plt.show()


# In[ ]:


df2.shape[0]
sns.pairplot(df2[['cwc_max' , 'csc_max' , 'ctc_max' ,'mean_ratio', 'is_duplicate']] , hue='is_duplicate' , vars = ['cwc_max' , 'csc_max' , 'ctc_max' ,'mean_ratio'])
plt.show()


# In[ ]:


df2.shape[0]
sns.pairplot(df2[['fuzz_ratio' , 'fuzz_partial_ratio' , 'token_set_ratio' , 'token_sort_ratio', 'is_duplicate']] , hue='is_duplicate' , vars=['fuzz_ratio' , 'fuzz_partial_ratio' , 'token_set_ratio' , 'token_sort_ratio', ])
plt.show()


# <h2>1.9 Plotting Violin Plot and Distributed Plot for feature Token sort ratio.

# In[ ]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
sns.violinplot(x='is_duplicate' , y='token_sort_ratio' , data = df2)

plt.subplot(1,2,2)
sns.distplot(df2[df2['is_duplicate'] == 0.0] ['token_sort_ratio'] , label = '0' , color = 'red' )
sns.distplot(df2[df2['is_duplicate'] == 1.0] ['token_sort_ratio'] , label = '1' , color = 'blue')
plt.show()


# <h2>1.10 Plotting Violin Plot and Distributed Plot for feature Longest common substring.

# In[ ]:


plt.figure(figsize=(15,8))
plt.subplot(1,2,1)
sns.violinplot(x='is_duplicate' , y='longest_common_substring' , data = df2)

plt.subplot(1,2,2)
sns.distplot(df2[df2['is_duplicate'] == 0.0] ['longest_common_substring'] , label = '0' , color = 'red' )
sns.distplot(df2[df2['is_duplicate'] == 1.0] ['longest_common_substring'] , label = '1' , color = 'blue')
plt.show()


# <h2>1.11 Scale the data between 0 and 1. For this we use MinMaxScaler

# In[ ]:


df.columns


# In[ ]:


df2.columns


# In[ ]:


# ust taking sample size.
df2_sample = df2[0:10000]
x = MinMaxScaler().fit_transform(df2_sample[['cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max','last_word_common', 'first_word_common', 'abs_len_diff', 'mean_ratio',
       'fuzz_ratio', 'fuzz_partial_ratio', 'token_set_ratio','token_sort_ratio', 'longest_common_substring']])
y = df2_sample['is_duplicate']


# <h2>1.12 Plotting using TSNE

# <h3>1.12.1 t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique commonly used for visualizing high-dimensional data in a lower-dimensional space. It is particularly useful for exploring and understanding complex patterns and structures in data.
# 
# <h3>The main idea behind t-SNE is to represent each data point as a two- or three-dimensional point on a scatter plot while preserving the local structure and relationships between data points from the original high-dimensional space. It accomplishes this by modeling the similarity between data points in both the high-dimensional and low-dimensional spaces.

# In[ ]:


# 2D TSNE
tsne_2d_per50 = TSNE(n_components = 2, verbose=2, init='random', perplexity= 50, n_iter=2000, random_state=100).fit_transform(x)


# In[ ]:


tsne_2d_per30 = TSNE(n_components = 2, verbose=2, init='random', perplexity= 30, n_iter=2000, random_state=100).fit_transform(x)


# In[ ]:


tsne_2d_per70 = TSNE(n_components = 2, verbose=2, init='random', perplexity= 70, n_iter=2000, random_state=100).fit_transform(x)


# In[ ]:


df = pd.DataFrame({'x': tsne_2d_per70[ :,0], 'y': tsne_2d_per70[ :,1], 'label':y})
sns.lmplot(data=df, x='x', y='y', hue='label', markers=['s','o'], fit_reg=False, height=5)
plt.title("Perplexity {} and n_iter {}".format(70,2000))
plt.show()


# <h2>1.13 Lets do some text Preprocessing with Tfidf Word Vector</h3>

# In[ ]:


df = pd.read_csv("/content/gdrive/My Drive/Quora/train.csv")
df['question1'] = df['question1'].apply(lambda x: str(x))
df['question2'] = df['question2'].apply(lambda x: str(x))
df.head()


# <h3>1.13.1 The below code segment calculates the TF-IDF values for the words present in the 'question1' and 'question2' columns of the DataFrame and creates a dictionary mapping each word to its IDF value.
# 

# In[ ]:


question = list(df['question1']) + list(df['question2'])
tfidf = TfidfVectorizer(lowercase=False)
tfidf.fit_transform(question)
word2tfidf = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))


# <h3>1.13.2 The below code snippet calculates the mean word vectors for each question in the 'question1' column of the DataFrame using word vectors provided by the spaCy model. It also incorporates IDF values from the word2tfidf dictionary to weight the word vectors before calculating the mean.

# In[ ]:


# the en_core_web_sm model from the spaCy library is loaded using spacy.load('en_core_web_sm'). This model is trained on English
# language text and provides word vectors and linguistic annotations.
nlp = spacy.load('en_core_web_sm')
vec1 = []
for qu1 in tqdm(list(df['question1'])):
  doc1 = nlp(qu1)
  mean_vect1 = np.zeros([len(doc1) , len(doc1[0].vector)])
  for word1 in doc1:
    vect1 = word1.vector
    try:
      idf = word2tfidf[str(word1)]
    except:
      idf = 0
    mean_vect1 += vect1 * idf
  mean_vect1 = mean_vect1.mean(axis=0)
  vec1.append(mean_vect1)
df['q1_feat_m'] = list(vec1)


# In[ ]:


vec2 = []
for qu2 in tqdm(list(df['question2'])):
  doc2 = nlp(qu2)
  mean_vect2 = np.zeros([len(doc2) , len(doc2[0].vector)])
  for word2 in doc2:
    vect2 = word2.vector
    try:
      idf = word2tfidf[str(word2)]
    except:
      idf = 0
    mean_vect2 += vect2 * idf
  mean_vect2 = mean_vect2.mean(axis=0)
  vec2.append(mean_vect2)
df['q2_feat_m'] = list(vec2)


# In[ ]:


# Load the dataset if present in the given location else compute it.
if os.path.isfile("/content/gdrive/My Drive/Quora/basic_feature_extraction.csv"):
  df_basic_feat = pd.read_csv("/content/gdrive/My Drive/Quora/basic_feature_extraction.csv", encoding='latin-1')
else:
  print("Compute the basic feature extraction.")

if os.path.isfile("/content/gdrive/My Drive/Quora/nlp_feature_train.csv"):
  df_nlp_feat = pd.read_csv("/content/gdrive/My Drive/Quora/nlp_feature_train.csv", encoding='latin-1')
else:
  print("Compute the nlp feature train.")


# In[ ]:


df_basic_feat.columns


# In[ ]:


df_nlp_feat.columns


# In[ ]:


# dropping bunch of columns for future use.
df1    = df_basic_feat.drop(['qid1','qid2','question1','question2'] , axis=1)
df2    = df_nlp_feat.drop(['qid1', 'qid2', 'question1', 'question2','is_duplicate'],axis = 1)
df3    = df.drop(['qid1', 'qid2', 'question1', 'question2','is_duplicate'],axis = 1)
df3_q1 = pd.DataFrame(df.q1_feat_m.values.tolist(), index=df3.index)
df3_q2 = pd.DataFrame(df.q2_feat_m.values.tolist(), index=df3.index)


# In[ ]:


print("Number of features in Basic Features Extraction: {}".format(df1.shape[1]))
df1.head(2)


# In[ ]:


print("Number of features in NLP Features Extraction: {}".format(df2.shape[1]))
df2.head(2)


# In[ ]:


print("Number of features in question1 W2V: {}".format(df3_q1.shape[1]))
df3_q1.head(2)


# In[ ]:


print("Number of features in question2 W2V: {}".format(df3_q2.shape[1]))
df3_q2.head(2)


# In[3]:


# Load the dataset if present in the drive else merge the above dataframes and create one final dataframe.
if os.path.isfile("/content/gdrive/My Drive/Quora/final_feature.csv"):
  final_df = pd.read_csv("/content/gdrive/My Drive/Quora/final_feature.csv")
  final_df.head()
else:
  df3_q1['id'] = df1['id']
  df3_q2['id'] = df1['id']
  df1 = df1.merge(df2 , on='id', how='left')
  df3_q1 = df3_q1.merge(df3_q2 , on='id', how='left')
  result = df1.merge(df3_q1 , on='id' , how='left')
  result.to_csv("/content/gdrive/My Drive/Quora/final_feature.csv")
  result.head()


# In[4]:


# if you get NameError then recompute the above cell.
final_df.head()


# In[ ]:


print("*"*20,"Columns in final data",20*"*")
for i in final_df.columns:
  print(i)


# <h2>1.14 Separating data for X's and Y's

# In[5]:


x = final_df.drop(['Unnamed: 0','id','is_duplicate'], axis=1)
y = final_df['is_duplicate']


# In[6]:


print(x.shape)
print(y.shape)


# <h2>1.15 Splitting data for Train(70%) and Test(30%)

# <h3>NOTE: You can split data into train, cross validate and test dataset into the ratio 70:15:15 respectivelly.

# In[7]:


# Splitting the dataset in train and test.
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.3, random_state= 1 )
print("Number of points in X-Train : {}".format(x_train.shape))
print("Number of points in Y-Train : {}".format(y_train.shape))
print("Number of points in X-Test : {}".format(x_test.shape))
print("Number of points in Y-Test : {}".format(y_test.shape))


# <h3>1.15.1 The below code provides insights into the class distribution of the training and test sets by calculating the percentage of data points belonging to each class.

# In[8]:


train_dist = Counter(y_train) # Counter output---> Counter({0: 178655, 1: 104348})
train_len  = len(y_train)
print("*"*20,"Distribution of Train Data Points","*"*20)
print("Class [0] contain",round((int(train_dist[0])/train_len)*100,2),"% of data" ," & Class [1] contain" ,round((int(train_dist[1])/train_len)*100,2), "% of data\n")

test_dist = Counter(y_test)
test_len  = len(y_test)
print("-"*80)
print("*"*20,"Distribution of Test Data Points","*"*20)
print("Class [0] contain",round((int(test_dist[0])/test_len)*100,2),"% of data" ," & Class [1] contain" ,round((int(test_dist[1])/test_len)*100,2), "% of data")


# <h3> 1.15.3 The below code snippet defines a function to plot the confusion matrix, precision matrix, and recall matrix based on the predicted and true labels. It provides visual representations of the performance metrics for a classification model.

# In[9]:


'''

                 Predicted Negative    Predicted Positive
Actual Negative        TN                      FP
Actual Positive        FN                      TP

To read and interpret a confusion matrix:
1. True Positive (TP): The number of correctly predicted positive instances.
2. True Negative (TN): The number of correctly predicted negative instances.
3. False Positive (FP): The number of instances that were actually negative but incorrectly predicted as positive.
4. False Negative (FN): The number of instances that were actually positive but incorrectly predicted as negative.

Here are some common evaluation metrics derived from a confusion matrix:
1. Accuracy: (TP + TN) / (TP + TN + FP + FN)
2. Precision: TP / (TP + FP)
3. Recall (Sensitivity): TP / (TP + FN)
4. Specificity: TN / (TN + FP)
5. F1-score: 2 * (Precision * Recall) / (Precision + Recall)

'''
def confussion_matrics(y_test , y_predict):
  C = confusion_matrix(y_test , y_predict)
  A = ((C.T/(C.sum(axis=1))).T) # precision
  B = (C/C.sum(axis=0)) # recall
  label= [1,2]
  plt.figure(figsize=(20,5))
  cmap = sns.light_palette("green")

  plt.subplot(1,3,1)
  sns.heatmap(C , annot=True, cmap=cmap, fmt=".3f", xticklabels=label , yticklabels=label)
  plt.xlabel("Predicted Values")
  plt.ylabel("Original Values")
  plt.title("Confussion Matrics")
  

  plt.subplot(1,3,2)
  sns.heatmap(A , annot=True, cmap=cmap, fmt=".3f", xticklabels=label , yticklabels=label)
  plt.xlabel("Predicted Values")
  plt.ylabel("Original Values")
  plt.title("Precision Matrics")
  

  plt.subplot(1,3,3)
  sns.heatmap(B , annot=True, cmap=cmap, fmt=".3f", xticklabels=label , yticklabels=label)
  plt.xlabel("Predicted Values")
  plt.ylabel("Original Values")
  plt.title("Recall Matrics")
  plt.show()


# <h2>1.16 Train with Models

# <h3>1.16.1 Random model (The Base Model)

# <h4>1.16.1.1  The random model provides a useful baseline for evaluating and comparing the performance of other models, acting as a benchmark for the task at hand. It helps us gauge whether our models are learning meaningful patterns and guides us in improving model performance.

# In[ ]:


predicted_y = np.zeros((test_len,2))
for i in range(test_len):
  rand_probs = np.random.rand(1,2)
  predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log Loss for Random model is ",log_loss(y_test , predicted_y , eps=1e-15))
predicted_y = np.argmax(predicted_y,axis=1)
confussion_matrics(y_test ,predicted_y )


# ***Observation:*** The logloss for base model is 0.8848137 approx. , so this is acting as the baseline for other good models. If any model whose performance is low then base model then we don't need to waste our time and resources on that.

#  <h3>1.16.2 Logistic Regression with L2 penalty (Ridge regularization)

# In[ ]:


alpha = [10 ** x for x in range(-5, 2)]
log_loss_error = []
for i in alpha:
  clf = SGDClassifier(alpha = i , penalty='l2', loss='log', random_state=40)
  clf.fit(x_train,y_train)
  sig_clf = CalibratedClassifierCV(clf, method='sigmoid')
  sig_clf.fit(x_train,y_train)
  predicted_y = sig_clf.predict_proba(x_test)
  log_loss_error.append(log_loss(y_test, predicted_y, labels=clf.classes_,eps=1e-15))
  print("For value of alpha ", i , " the log loss is",log_loss(y_test, predicted_y, labels=clf.classes_, eps=1e-15))

fig, ax = plt.subplots()
ax.plot(alpha , log_loss_error, c='b')
for i, txt in enumerate(np.round(log_loss_error,3)):
  ax.annotate((alpha[i], np.round(txt,3)),(alpha[i] , log_loss_error[i]))
plt.grid()
plt.xlabel("Alpha's")
plt.ylabel("Log Loss")
plt.title("Log Loss for Logistic Regression")
plt.show()

best_alpha = np.argmin(log_loss_error)
clf = SGDClassifier(alpha=best_alpha, penalty="l2", loss="log", random_state=40)
clf.fit(x_train,y_train)
sig_clf = CalibratedClassifierCV(clf, method='sigmoid')
sig_clf.fit(x_train,y_train)

predicted_y = sig_clf.predict_proba(x_train)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predicted_y, labels=clf.classes_, eps=1e-15))
predicted_y = sig_clf.predict_proba(x_test)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predicted_y, labels=clf.classes_, eps=1e-15))
predicted_y =np.argmax(predicted_y,axis=1)
print("Total number of data points :", len(predicted_y))
confussion_matrics(y_test, predicted_y)


# ***Observation***: Log Reg with L2 penalty performs significantly better on alpha= 10 than our base model with train logloss of 0.627819 and test logloss of 0.628503. Here the both train and test logloss are close to each other which means our model is not underfitting or overfitting. 

# <h3>1.16.3 Logistic Regression with L1 penalty (Lasso regularization) 

# In[ ]:


# L1 penalty creates sparse matrix.
# We can change CalibratedClassifierCV method to 'isotonic' to get the different result.
alpha = [10 ** x for x in range(-5, 2)]
log_loss_error = []
for i in alpha:
  clf = SGDClassifier(alpha = i , penalty='l1', loss='log', random_state=40)
  clf.fit(x_train,y_train)
  sig_clf = CalibratedClassifierCV(clf, method='sigmoid')
  sig_clf.fit(x_train,y_train)
  predicted_y = sig_clf.predict_proba(x_test)
  log_loss_error.append(log_loss(y_test, predicted_y, labels=clf.classes_,eps=1e-15))
  print("For value of alpha ", i , " the log loss is",log_loss(y_test, predicted_y, labels=clf.classes_, eps=1e-15))

fig, ax = plt.subplots()
ax.plot(alpha , log_loss_error, c='b')
for i, txt in enumerate(np.round(log_loss_error,3)):
  ax.annotate((alpha[i], np.round(txt,3)),(alpha[i] , log_loss_error[i]))
plt.grid()
plt.xlabel("Alpha's")
plt.ylabel("Log Loss")
plt.title("Log Loss for Logistic Regression")
plt.show()


best_alpha = np.argmin(log_loss_error)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty="l1", loss="log", random_state=40)
clf.fit(x_train,y_train)
sig_clf = CalibratedClassifierCV(clf, method='sigmoid')
sig_clf.fit(x_train,y_train)

predicted_y = sig_clf.predict_proba(x_train)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predicted_y, labels=clf.classes_, eps=1e-15))
predicted_y = sig_clf.predict_proba(x_test)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predicted_y, labels=clf.classes_, eps=1e-15))
predicted_y =np.argmax(predicted_y,axis=1)
print("Total number of data points :", len(predicted_y))
confussion_matrics(y_test, predicted_y)


# ***Observation***: Log Reg with L1 penalty performs significantly better on alpha= 0.1 than our base model with train logloss of 0.5491409 and test logloss of 0.5491409. Here the both train and test logloss are close to each other which means our model is not underfitting or overfitting. 

# <h3>1.16.4 Linear Regression with hyperparameter tunning
# 

# In[ ]:


alpha = [10 ** x for x in range(-5, 2)] # hyperparam for SGD classifier.
log_error_array=[]
for i in alpha:
    clf = SGDClassifier(alpha=i, penalty='l1', loss='hinge', random_state=42)
    clf.fit(x_train, y_train)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(x_train, y_train)
    predicted_y = sig_clf.predict_proba(x_test)
    log_error_array.append(log_loss(y_test, predicted_y, labels=clf.classes_, eps=1e-15))
    print('For values of alpha = ', i, "The log loss is:",log_loss(y_test, predicted_y, labels=clf.classes_, eps=1e-15))

fig, ax = plt.subplots()
ax.plot(alpha, log_error_array,c='g')
for i, txt in enumerate(np.round(log_error_array,3)):
    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],log_error_array[i]))
plt.grid()
plt.title("Cross Validation Error for each alpha")
plt.xlabel("Alpha i's")
plt.ylabel("Error measure")
plt.show()


best_alpha = np.argmin(log_error_array)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l1', loss='hinge', random_state=42)
clf.fit(x_train, y_train)
sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
sig_clf.fit(x_train, y_train)

predict_y = sig_clf.predict_proba(x_train)
print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))
predict_y = sig_clf.predict_proba(x_test)
print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
predicted_y =np.argmax(predict_y,axis=1)
print("Total number of data points :", len(predicted_y))
confussion_matrics(y_test, predicted_y)


# ***Observation:*** Linear Regression with some hyperparameter tunning get the train logloss of 0.49734905 and test logloss of 0.49734905 with alpha =0.01. Hence Linear Regression performs better than Logistic Regression and no underfitting and overfitting is observed.

# <h3>1.16.5 Modeling XG Boost Classifier

# 1.16.5.1 The code below trains an XGBoost model, evaluates its performance using log loss, performs hyperparameter tuning through randomized search, and visualizes the confusion matrix for the predicted labels.

# In[18]:


import xgboost as xgb
from scipy.stats import randint, uniform

learning_rate = uniform(loc=0.01, scale=0.1 - 0.01).rvs()
max_depth = randint(3, 6).rvs()

# Define the parameter grid
param_distributions = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': learning_rate,
    'max_depth': max_depth,
}
d_train = xgb.DMatrix(x_train, label=y_train)
d_test = xgb.DMatrix(x_test, label=y_test)

watchlist = [(d_train, 'train'), (d_test, 'valid')]

bst = xgb.train(param_distributions, d_train, num_boost_round=100, evals=watchlist,
                early_stopping_rounds=20, verbose_eval=10)

predicted_y = bst.predict(d_test)
print("The test log loss is:", log_loss(y_test, predicted_y))


# <h4> This can be more reduce with better hyperparameter tunning.

# In[ ]:


param_distributions = [{'max_depth': list(range(2,6)), 'max_features': list(range(0,15))} ]
model = XGBClassifier(n_jobs=-1)
model.fit(x_train,y_train)
rand_clf = RandomizedSearchCV(model, n_jobs = -1, param_distributions=param_distributions)
rand_clf.fit(x_train,y_train)
predicted_y = rand_clf.predict_proba(x_train)
print("Log loss on train data is : ",log_loss(y_train, predicted_y, labels=clf.classes_, eps=1e-5))
predicted_y = rand_clf.predict_proba(x_test)
print("Log loss on test data is : ",log_loss(y_test, predicted_y, labels=clf.classes_, eps=1e-5))
confussion_matrics(y_test,predicted_y)


# ***Observation:***

# <h3>1.16.6 Modelling with Decision Tress with Gini.

# 1.16.6.1  The below code trains a decision tree classifier, evaluates its performance using log loss, and visualizes the confusion matrix for the predicted labels. The log loss measures the accuracy of the predicted probabilities, and the confusion matrix provides insights into the model's classification performance.
# 
# The Gini impurity is calculated for each potential split point in the decision tree. The split that minimizes the Gini impurity is chosen as the best split, as it results in the greatest purity or homogeneity(all samples belong to the same class) in the resulting child nodes.

# In[ ]:


model = DecisionTreeClassifier(criterion='gini', max_depth=3 , min_samples_split=8 , min_samples_leaf=4 , random_state=100)
model.fit(x_train , y_train)
predicted_y = model.predict(x_train)
print("Log loss for Train: ",log_loss(y_train , predicted_y , labels = model.classes_ , eps = 1e-10))
predicted_y = model.predict(x_test)
print("Log loss for Test: " ,log_loss(y_test , predicted_y , labels = model.classes_ , eps = 1e-10))

confussion_matrics(y_test , predicted_y)


# ***Observation:*** Decision Tree with gini impurity seems disaster.

# <h3>1.16.7 Modelling with Decision Tree with Entropy
# 

# In[ ]:


from scipy.stats import randint
param_distributions = {
    'max_depth': randint(low=1, high=10),  # Random integer values between 1 and 10
    'criterion': ['gini', 'entropy'],      # Choice between 'gini' and 'entropy'
    'random_state': [100]                   # Fixed value for random_state
}

model = DecisionTreeClassifier()
rand_cv = RandomizedSearchCV(model, param_distributions=param_distributions, random_state=0)
rand_cv.fit(x_train, y_train)
predicted_y = rand_cv.predict_proba(x_train)
print("Train Log Loss: ", log_loss(y_train, predicted_y, labels=rand_cv.classes_, eps=1e-10))
predicted_y = rand_cv.predict_proba(x_test)
print("Test Log Loss: ", log_loss(y_test, predicted_y, labels=rand_cv.classes_, eps=1e-10))
confussion_matrics(y_test, np.argmax(predicted_y, axis=1))


# ***Observation:*** This model perform much better than decision tree with gini inpurity.
