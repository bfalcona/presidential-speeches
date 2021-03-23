#!/usr/bin/env python
# coding: utf-8

# Universal stuff

# In[7]:


import string
import nltk
import gensim
import numpy as np
from pathlib import Path

stopwords = nltk.corpus.stopwords.words("english")

lemmatizer = nltk.stem.WordNetLemmatizer()

def penntag_to_wordnettag(tag):
    if tag.startswith("NN"):
        return nltk.corpus.wordnet.NOUN
    elif tag.startswith("VB"):
        return nltk.corpus.wordnet.VERB
    elif tag.startswith("JJ"):
        return nltk.corpus.wordnet.ADJ
    elif tag.startswith("RB"):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN
    
common_terms = ["America", "American", "Americans", "Applause.", "Applause", "applause.", "applause", "back",
                "Congress", "could", "country", "good", "government", "know", "like", "make", "must", "nation",
                "people", "President", "president", "United", "States", "They", "they", "this", "want", "would",
                "year","great", "national", "need", "many", "well", "take", "this", "trump", "think", "happen",
                "vote", "world", "time", "come", "life", "look", "never", "This", "upon", "purpose", "shall", "goal",
                "first", "every", "work", "help", "today", "meet", "tonight", "federal", "last", "right", "tell",
                "thing", "that", "much", "south", "South", "also", "believe", "north", "North", "east", "East",
                "program", "state", "policy", "increase", "thank", "give", "percent", "booing", "decision", "problem",
                "well", "begin", "APPLAUSE", "TRUMP", "HILLARY", "CLINTON", "office", "Well", "Hillary", "Clinton",
                "Trump", "audience", "bring", "BOOING", "Thank", "That", "that", "Ever", "ever", "AUDIENCE", "Even",
                "even", "continue", "folk", "Folk", "leave", "Leave", "talk", "Talk", "Lose", "lose", "Really",
                "really", "Million", "million", "Care", "care", "Place", "place", "mean", "Mean", "Love", "love",
                "Start", "start", "city"]


# Dwight D. Eisenhower

# In[2]:


eisenhower_path = Path("presidential_speeches/eisenhower")

eisenhower_files = [ ]
for file in eisenhower_path.iterdir():
    if file.name != ".DS_Store":
        eisenhower_files.append(file)

with open("output_file", "w") as outfile:
    for fname in eisenhower_files:
        with open(fname) as infile:
            outfile.write(infile.read())
            
with open("output_file") as f:
    eisenhower_uncleaned = f.read()

eisenhower_tagged = nltk.pos_tag(nltk.word_tokenize(eisenhower_uncleaned))

eisenhower_lemmatized = [ ]
for word, tag in eisenhower_tagged:
    wntag = penntag_to_wordnettag(tag)
    eisenhower_lemmatized.append(lemmatizer.lemmatize(word, wntag))

eisenhower_text = [ ]
for word in eisenhower_lemmatized:
    if word not in stopwords and word not in common_terms and len(word) >= 4 and word.strip(string.punctuation) != "":
        eisenhower_text.append(word.lower())

eisenhower_corpus = [ ]
eisenhower_corpus.append(eisenhower_text)

eisenhower_dictionary = gensim.corpora.Dictionary(eisenhower_corpus)
eisenhower_gensim = [ ]
for document in eisenhower_corpus:
    eisenhower_gensim.append(eisenhower_dictionary.doc2bow(document))

eisenhower_model = gensim.models.ldamodel.LdaModel(eisenhower_gensim,
                                               num_topics = 5,
                                               id2word = eisenhower_dictionary,
                                               passes = 20, iterations = 500,
                                               alpha = "asymmetric",
                                               random_state = 0)

for topic in eisenhower_model.print_topics(num_words = 10):
    print(topic)


# Richard Nixon

# In[8]:


dtype = np.float64

nixon_path = Path("presidential_speeches/nixon")

nixon_files = [ ]
for file in nixon_path.iterdir():
    if file.name != ".DS_Store":
        nixon_files.append(file)

with open("output_file", "w") as outfile:
    for fname in nixon_files:
        with open(fname) as infile:
            outfile.write(infile.read())
            
with open("output_file") as f:
    nixon_uncleaned = f.read()

nixon_tagged = nltk.pos_tag(nltk.word_tokenize(nixon_uncleaned))

nixon_lemmatized = [ ]
for word, tag in nixon_tagged:
    wntag = penntag_to_wordnettag(tag)
    nixon_lemmatized.append(lemmatizer.lemmatize(word, wntag))

nixon_text = [ ]
for word in nixon_lemmatized:
    if word not in stopwords and word not in common_terms and len(word) >= 4 and word.strip(string.punctuation) != "":
        nixon_text.append(word.lower())

nixon_corpus = [ ]
nixon_corpus.append(nixon_text)

nixon_dictionary = gensim.corpora.Dictionary(nixon_corpus)
nixon_gensim = [ ]
for document in nixon_corpus:
    nixon_gensim.append(nixon_dictionary.doc2bow(document))

nixon_model = gensim.models.ldamodel.LdaModel(nixon_gensim,
                                               num_topics = 5,
                                               id2word = nixon_dictionary,
                                               passes = 20, iterations = 500,
                                               alpha = "asymmetric",
                                               random_state = 0)

for topic in nixon_model.print_topics(num_words = 10):
    print(topic)


# Gerald Ford

# In[4]:


ford_path = Path("presidential_speeches/ford")

ford_files = [ ]
for file in ford_path.iterdir():
    if file.name != ".DS_Store":
        ford_files.append(file)

with open("output_file", "w") as outfile:
    for fname in ford_files:
        with open(fname) as infile:
            outfile.write(infile.read())
            
with open("output_file") as f:
    ford_uncleaned = f.read()

ford_tagged = nltk.pos_tag(nltk.word_tokenize(ford_uncleaned))

ford_lemmatized = [ ]
for word, tag in ford_tagged:
    wntag = penntag_to_wordnettag(tag)
    ford_lemmatized.append(lemmatizer.lemmatize(word, wntag))

ford_text = [ ]
for word in ford_lemmatized:
    if word not in stopwords and word not in common_terms and len(word) >= 4 and word.strip(string.punctuation) != "":
        ford_text.append(word.lower())

ford_corpus = [ ]
ford_corpus.append(ford_text)

ford_dictionary = gensim.corpora.Dictionary(ford_corpus)
ford_gensim = [ ]
for document in ford_corpus:
    ford_gensim.append(ford_dictionary.doc2bow(document))

ford_model = gensim.models.ldamodel.LdaModel(ford_gensim,
                                               num_topics = 5,
                                               id2word = ford_dictionary,
                                               passes = 20, iterations = 500,
                                               alpha = "asymmetric",
                                               random_state = 0)

for topic in ford_model.print_topics(num_words = 10):
    print(topic)


# Ronald Reagan

# In[5]:


reagan_path = Path("presidential_speeches/reagan")

reagan_files = [ ]
for file in reagan_path.iterdir():
    if file.name != ".DS_Store":
        reagan_files.append(file)

with open("output_file", "w") as outfile:
    for fname in reagan_files:
        with open(fname) as infile:
            outfile.write(infile.read())
            
with open("output_file") as f:
    reagan_uncleaned = f.read()

reagan_tagged = nltk.pos_tag(nltk.word_tokenize(reagan_uncleaned))

reagan_lemmatized = [ ]
for word, tag in reagan_tagged:
    wntag = penntag_to_wordnettag(tag)
    reagan_lemmatized.append(lemmatizer.lemmatize(word, wntag))

reagan_text = [ ]
for word in reagan_lemmatized:
    if word not in stopwords and word not in common_terms and len(word) >= 4 and word.strip(string.punctuation) != "":
        reagan_text.append(word.lower())

reagan_corpus = [ ]
reagan_corpus.append(reagan_text)

reagan_dictionary = gensim.corpora.Dictionary(reagan_corpus)
reagan_gensim = [ ]
for document in reagan_corpus:
    reagan_gensim.append(reagan_dictionary.doc2bow(document))

reagan_model = gensim.models.ldamodel.LdaModel(reagan_gensim,
                                               num_topics = 5,
                                               id2word = reagan_dictionary,
                                               passes = 20, iterations = 500,
                                               alpha = "asymmetric",
                                               random_state = 0)

for topic in reagan_model.print_topics(num_words = 10):
    print(topic)


# George H. W. Bush

# In[6]:


bush_path = Path("presidential_speeches/bush")

bush_files = [ ]
for file in bush_path.iterdir():
    if file.name != ".DS_Store":
        bush_files.append(file)
bush_files

with open("output_file", "w") as outfile:
    for fname in bush_files:
        with open(fname) as infile:
            outfile.write(infile.read())
            
with open("output_file") as f:
    bush_uncleaned = f.read()

bush_tagged = nltk.pos_tag(nltk.word_tokenize(bush_uncleaned))

bush_lemmatized = [ ]
for word, tag in bush_tagged:
    wntag = penntag_to_wordnettag(tag)
    bush_lemmatized.append(lemmatizer.lemmatize(word, wntag))

bush_text = [ ]
for word in bush_lemmatized:
    if word not in stopwords and word not in common_terms and len(word) >= 4 and word.strip(string.punctuation) != "":
        bush_text.append(word.lower())

bush_corpus = [ ]
bush_corpus.append(bush_text)

bush_dictionary = gensim.corpora.Dictionary(bush_corpus)
bush_gensim = [ ]
for document in bush_corpus:
    bush_gensim.append(bush_dictionary.doc2bow(document))
bush_gensim

bush_model = gensim.models.ldamodel.LdaModel(bush_gensim,
                                               num_topics = 5,
                                               id2word = bush_dictionary,
                                               passes = 20, iterations = 500,
                                               alpha = "asymmetric",
                                               random_state = 0)

for topic in bush_model.print_topics(num_words = 10):
    print(topic)


# George W. Bush

# In[7]:


gwbush_path = Path("presidential_speeches/gwbush")

gwbush_files = [ ]
for file in gwbush_path.iterdir():
    if file.name != ".DS_Store":
        gwbush_files.append(file)
gwbush_files

with open("output_file", "w") as outfile:
    for fname in gwbush_files:
        with open(fname) as infile:
            outfile.write(infile.read())
            
with open("output_file") as f:
    gwbush_uncleaned = f.read()

gwbush_tagged = nltk.pos_tag(nltk.word_tokenize(gwbush_uncleaned))

gwbush_lemmatized = [ ]
for word, tag in gwbush_tagged:
    wntag = penntag_to_wordnettag(tag)
    gwbush_lemmatized.append(lemmatizer.lemmatize(word, wntag))

gwbush_text = [ ]
for word in gwbush_lemmatized:
    if word not in stopwords and word not in common_terms and len(word) >= 4 and word.strip(string.punctuation) != "":
        gwbush_text.append(word.lower())

gwbush_corpus = [ ]
gwbush_corpus.append(gwbush_text)

gwbush_dictionary = gensim.corpora.Dictionary(gwbush_corpus)
gwbush_gensim = [ ]
for document in gwbush_corpus:
    gwbush_gensim.append(gwbush_dictionary.doc2bow(document))
gwbush_gensim

gwbush_model = gensim.models.ldamodel.LdaModel(gwbush_gensim,
                                               num_topics = 5,
                                               id2word = gwbush_dictionary,
                                               passes = 20, iterations = 500,
                                               alpha = "asymmetric",
                                               random_state = 0)

for topic in gwbush_model.print_topics(num_words = 10):
    print(topic)


# Donald Trump

# In[8]:


trump_path = Path("presidential_speeches/trump")

trump_files = [ ]
for file in trump_path.iterdir():
    if file.name != ".DS_Store":
        trump_files.append(file)
trump_files

with open("output_file", "w") as outfile:
    for fname in trump_files:
        with open(fname) as infile:
            outfile.write(infile.read())
            
with open("output_file") as f:
    trump_uncleaned = f.read()
    
trump_tagged = nltk.pos_tag(nltk.word_tokenize(trump_uncleaned))

trump_lemmatized = [ ]
for word, tag in trump_tagged:
    wntag = penntag_to_wordnettag(tag)
    trump_lemmatized.append(lemmatizer.lemmatize(word, wntag))

trump_text = [ ]
for word in trump_lemmatized:
    if word not in stopwords and word not in common_terms and len(word) >= 4 and word.strip(string.punctuation) != "":
        trump_text.append(word.lower())

trump_corpus = [ ]
trump_corpus.append(trump_text)

trump_dictionary = gensim.corpora.Dictionary(trump_corpus)
trump_gensim = [ ]
for document in trump_corpus:
    trump_gensim.append(trump_dictionary.doc2bow(document))
trump_gensim

trump_model = gensim.models.ldamodel.LdaModel(trump_gensim,
                                               num_topics = 5,
                                               id2word = trump_dictionary,
                                               passes = 20, iterations = 500,
                                               alpha = "asymmetric",
                                               random_state = 0)

for topic in trump_model.print_topics(num_words = 10):
    print(topic)

