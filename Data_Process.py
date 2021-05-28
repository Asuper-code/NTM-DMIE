#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#English wordPreprocessing
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
def wordProcess(sentence):
    """
    1.tokenize
    2.words_lower
    3.pos_tag
    4.stemming
    5.remove stopwords 
    input:  sentence string     ;        output: cleaned tokens
    """
    token_words = word_tokenize(sentence)          #word_tokenize
    token_words = [x.lower() for x in token_words] #lowercase
    token_tag = pos_tag(token_words)               #pos_tag
    words_lematizer = []                           #stemming
    wordnet_lematizer = WordNetLemmatizer()
    for word, tag in token_tag:
        if tag.startswith('NN'):
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='n')   
        elif tag.startswith('VB'): 
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='v')    
        elif tag.startswith('JJ'): 
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='a')
        elif tag.startswith('R'): 
            word_lematizer =  wordnet_lematizer.lemmatize(word, pos='r')            
        else: 
            word_lematizer =  wordnet_lematizer.lemmatize(word)
        words_lematizer.append(word_lematizer)
    cleaned_words = [word for word in words_lematizer if word not in stopwords.words('english')] #stopwords
    characters = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','-','...','^','{','}']
    words_list = [word for word in cleaned_words if word not in characters] 
    return words_list

#####laleTransExample
def labelTrans(label):
    for i in range(len(label)):
        if label[i] =="1":
            label[i] = 0
        if label[i] =="0":
            label[i] = 1
        if label[i]=="-1":
            label[i]=2
    batch_size = 128
    class_num=3
    label = np.array(label)
    label = torch.from_numpy(label)
    label = label.view(128,1)
    one_hot = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
    return one_hot


########tokenize
import re
def tokenize(text):

    try:
        text = text.decode('utf-8').lower()
    except:
        text = text.encode('utf-8').decode('utf-8').lower()
    text = re.sub(u"\u2019|\u2018", "\'", text)
    text = re.sub(u"\u201c|\u201d", "\"", text)
    text = re.sub(r"http[s]?:[^\ ]+", " ", text)
    text = re.sub(r"&gt;", " ", text)
    text = re.sub(r"&lt;", " ", text)
    text = re.sub(r"&quot;", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"#\ ", "#", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"[\(\)\[\]\{\}]", r" ", text)
    text = re.sub(r"#", " #", text)
    text = re.sub(r"\@", " \@", text)
    text = re.sub(r"[\!\?\.\,\+\-\$\%\^\>\<\=\:\;\*\(\)\{\}\[\]\/\~\&\'\|]", " ", text)

    words = text.split()
    return words

#####clean for space
import re
def clean_space(text):
    match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i,new_i)
    return text
