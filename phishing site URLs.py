#!/usr/bin/env python
# coding: utf-8

# # importing some useful libraries

# In[ ]:


import pandas as pd # use for data manipulation and analysis
import numpy as np # use for multi-dimensional array and matrix

import seaborn as sns # use for high-level interface for drawing attractive and informative statistical graphics 
import matplotlib.pyplot as plt # It provides an object-oriented API for embedding plots into applications
get_ipython().run_line_magic('matplotlib', 'inline')
# It sets the backend of matplotlib to the 'inline' backend:
import time # calculate time 

from sklearn.linear_model import LogisticRegression # algo use to predict good or bad
from sklearn.naive_bayes import MultinomialNB # nlp algo use to predict good or bad

from sklearn.model_selection import train_test_split # spliting the data between feature and target
from sklearn.metrics import classification_report # gives whole report about metrics (e.g, recall,precision,f1_score,c_m)
from sklearn.metrics import confusion_matrix # gives info about actual and predict
from nltk.tokenize import RegexpTokenizer # regexp tokenizers use to split words from text  
from nltk.stem.snowball import SnowballStemmer # stemmes words
from sklearn.feature_extraction.text import CountVectorizer # create sparse matrix of words using regexptokenizes  
from sklearn.pipeline import make_pipeline # use for combining all prerocessors techniuqes and algos

from PIL import Image # getting images in notebook
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator# creates words colud

from bs4 import BeautifulSoup # use for scraping the data from website
from selenium import webdriver # use for automation chrome 
import networkx as nx # for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.

import pickle# use to dump model 

import warnings # ignores pink warnings 
warnings.filterwarnings('ignore')


# # Did some surfing and found some websites offering malicious links. And found some datasets

# In[ ]:


phishing_data1 = pd.read_csv('phishing_urls.csv',usecols=['domain','label'],encoding='latin1', error_bad_lines=False)
phishing_data1.columns = ['URL','Label']
phishing_data2 = pd.read_csv('phishing_data.csv')
phishing_data2.columns = ['URL','Label']
phishing_data3 = pd.read_csv('phishing_data2.csv')
phishing_data3.columns = ['URL','Label']


# In[ ]:


for l in range(len(phishing_data1.Label)):
    if phishing_data1.Label.loc[l] == '1.0':
        phishing_data1.Label.loc[l] = 'bad'
    else:
        phishing_data1.Label.loc[l] = 'good'


# # Concatenate All datasets in one.

# In[ ]:


frames = [phishing_data1, phishing_data2, phishing_data3]
phishing_urls = pd.concat(frames)


# In[ ]:


#saving dataset
phishing_urls.to_csv(r'phishing_site_urls.csv', index = False)


# # Loading the main dataset.

# In[35]:


phish_data = pd.read_csv('phishing_site_urls.csv')


# In[36]:


phish_data.head()


# In[37]:


phish_data.tail()


# In[38]:


phish_data.info()


# In[39]:


phish_data.isnull().sum()


# # Since it is classification problems so let's see the classes are balanced or imbalances

# In[40]:


import pandas as pd


# In[41]:


label_counts = pd.DataFrame(phish_data.Label.value_counts())


# In[42]:


import seaborn as sns


# In[43]:


sns.set_style('darkgrid')
sns.barplot(x=label_counts.index, y=label_counts.Label)


# # RegexpTokenizer

# In[44]:


from nltk.tokenize import RegexpTokenizer


# In[45]:


tokenizer = RegexpTokenizer(r'[A-Za-z]+')


# In[46]:


tokenizer.tokenize(phish_data.URL[0])


# In[47]:


phish_data.URL[0]


# In[49]:


import time


# In[50]:


print('Getting words tokenized ...')
t0= time.perf_counter()
phish_data['text_tokenized'] = phish_data.URL.map(lambda t: tokenizer.tokenize(t)) # doing with all rows
t1 = time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[51]:


phish_data.sample(5)


# # SnowballStemmer

# In[52]:


from nltk.stem import SnowballStemmer


# In[53]:


stemmer = SnowballStemmer("english")


# In[54]:


print('Getting words stemmed ...')
t0= time.perf_counter()
phish_data['text_stemmed'] = phish_data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[55]:


phish_data.sample(5)


# In[56]:


print('Getting joiningwords ...')
t0= time.perf_counter()
phish_data['text_sent'] = phish_data['text_stemmed'].map(lambda l: ' '.join(l))
t1= time.perf_counter() - t0
print('Time taken',t1 ,'sec')


# In[57]:


phish_data.sample(5)


# # Visualization

# In[58]:


bad_sites = phish_data[phish_data.Label == 'bad']
good_sites = phish_data[phish_data.Label == 'good']


# In[59]:


bad_sites.head()


# In[60]:


good_sites.head()


# # create a function to visualize the important keys from url

# In[47]:


get_ipython().system('pip install wordcloud')


# In[61]:


def plot_wordcloud(text, mask=None, max_words=400, max_font_size=400, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'com','http'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()


# In[62]:


print(good_sites.columns)


# In[63]:


data = good_sites.text_sent
data.reset_index(drop=True, inplace=True)


# In[64]:


import numpy as np


# In[65]:


from PIL import Image


# In[66]:


import matplotlib.pyplot as plt


# In[67]:


from wordcloud import WordCloud, STOPWORDS

common_text = str(data)
common_mask = np.array(Image.open('star.png'))
plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=120, 
               title = 'Most common words use in good urls', title_size=15)


# In[68]:


data = bad_sites.text_sent
data.reset_index(drop=True, inplace=True)


# In[69]:


common_text = str(data)
common_mask = np.array(Image.open('comment.png'))
plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=400, 
               title = 'Most common words use in bad urls', title_size=15)


# # CountVectorizer

# In[142]:


from sklearn.feature_extraction.text import CountVectorizer


# In[143]:


cv = CountVectorizer()


# In[144]:


#help(CountVectorizer())


# In[145]:


feature = cv.fit_transform(phish_data.text_sent) #transform all text which we tokenize and stemed


# In[146]:


feature[:5].toarray() # convert sparse matrix into array to print transformed features


# # Spliting the data

# In[191]:


from sklearn.model_selection import train_test_split


# In[192]:


trainX, testX, trainY, testY = train_test_split(feature, phish_data.Label)


# # LogisticRegression

# In[193]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()


# In[194]:


lr.fit(trainX,trainY)


# In[195]:


lr.score(testX,testY)


# In[152]:


Scores_ml = {}
Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)


# In[154]:


from sklearn.metrics import confusion_matrix


# In[156]:


from sklearn.metrics import classification_report


# In[157]:


print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# # MultinomialNB

# In[159]:


from sklearn.naive_bayes import MultinomialNB


# In[160]:


# create mnb object
mnb = MultinomialNB()


# In[161]:


mnb.fit(trainX,trainY)


# In[162]:


mnb.score(testX,testY)


# In[163]:


Scores_ml['MultinomialNB'] = np.round(mnb.score(testX,testY),2)


# In[164]:


print('Training Accuracy :',mnb.score(trainX,trainY))
print('Testing Accuracy :',mnb.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(mnb.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(mnb.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[166]:


acc = pd.DataFrame.from_dict(Scores_ml,orient = 'index',columns=['Accuracy'])
sns.set_style('darkgrid')
sns.barplot(x=acc.index, y=acc.Accuracy)


# In[168]:


from sklearn.pipeline import make_pipeline


# In[169]:


pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())
##(r'\b(?:http|ftp)s?://\S*\w|\w+|[^\w\s]+') ([a-zA-Z]+)([0-9]+) -- these tolenizers giving me low accuray 


# In[170]:


trainX, testX, trainY, testY = train_test_split(phish_data.URL, phish_data.Label)


# In[172]:


import warnings # ignores pink warnings 
warnings.filterwarnings('ignore')


# In[173]:


pipeline_ls.fit(trainX,trainY)


# In[182]:


pipeline_ls.score(testX,testY)


# In[183]:


print('Training Accuracy :',pipeline_ls.score(trainX,trainY))
print('Testing Accuracy :',pipeline_ls.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])


print('\nCLASSIFICATION REPORT\n')
print(classification_report(pipeline_ls.predict(testX), testY,
                            target_names =['Bad','Good']))

print('\nCONFUSION MATRIX')
plt.figure(figsize= (6,4))
sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")


# In[185]:


import pickle


# In[186]:


pickle.dump(pipeline_ls,open('phishing.pkl','wb'))


# In[187]:


loaded_model = pickle.load(open('phishing.pkl', 'rb'))
result = loaded_model.score(testX,testY)
print(result)


# In[ ]:


* Bad links => this are phishing sites
yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php
fazan-pacir.rs/temp/libraries/ipad
www.tubemoviez.exe
svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt

* Good links => this are not phishing sites
www.youtube.com/
youtube.com/watch?v=qI0TQJI3vdU
www.retailhellunderground.com/
restorevisioncenters.com/html/technology.html


# In[188]:


predict_bad = ['yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php','fazan-pacir.rs/temp/libraries/ipad','tubemoviez.exe','svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt']
predict_good = ['youtube.com/','youtube.com/watch?v=qI0TQJI3vdU','retailhellunderground.com/','restorevisioncenters.com/html/technology.html']
loaded_model = pickle.load(open('phishing.pkl', 'rb'))
#predict_bad = vectorizers.transform(predict_bad)
# predict_good = vectorizer.transform(predict_good)
result = loaded_model.predict(predict_bad)
result2 = loaded_model.predict(predict_good)
print(result)
print("*"*30)
print(result2)


# In[ ]:




