#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:32:52 2021

@author: jinnieshin
"""

import nltk 
from pprint import pprint
from nltk.corpus import wordnet as wn

# Just to make it a bit more readable
WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'


def convert(word, from_pos, to_pos):    
    """ Transform words given from/to POS tags """

    synsets = wn.synsets(word, pos=from_pos)

    # Word not found
    if not synsets:
        return []

    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = []
    for s in synsets:
        for l in s.lemmas():
            if s.name().split('.')[1] == from_pos or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and s.name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                lemmas += [l]

    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]

    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = []

    for drf in derivationally_related_forms:
        for l in drf[1]:
            if l.synset().name().split('.')[1] == to_pos or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE) and l.synset().name().split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE):
                related_noun_lemmas += [l]

    # Extract the words from the lemmas
    words = [l.name() for l in related_noun_lemmas]
    len_words = len(words)

    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w)) / len_words) for w in set(words)]
    result.sort(key=lambda w:-w[1])

    # return all the possibilities sorted by probability
    return result

def pos(word):
    tmp = wn.synsets(word)[0].pos()
    return tmp
    
def sent_score(text, topicA):
    sent_score = []
    for i in nltk.sent_tokenize(text):
        pnt =0
        for word in nltk.word_tokenize(i.lower()):
            if word in wordmap[topicA-1].keys(): 
                print(word)
                pnt += wordmap[topicA-1][word]
            else:
                pnt = pnt 
        sent_score.append(pnt)

    dic = dict(list(zip(nltk.sent_tokenize(text), sent_score)))
    score = [v for i, v in dic.items() if v!=0]
    dic = [(i,v) for i, v in dic.items() if v!=0]
    return dic, score
    
def item_model1(data = coherent, df_wrd=df_wrd, num= i):
    
    text = data.samples.iloc[num]
    try:
        text = clean_text(text)
    except:
        text = text
       
    q=[]
    topic_index =data.drop(columns=['total', 'samples', 'dominant', 'length']).iloc[num]
    topicA = topic_index.argmax()+1
    print('---- Topic A is Topic: ', topicA)
    words = df_wrd.iloc[topicA-1]
    keyword_list = words[words!=0].index
    #df_wrd.iloc[topicA-1].sort_values().index
    
    
    for i in [k for k in keyword_list if k in nltk.word_tokenize(text.lower())]:
        topicAkeyword = i
        stem= "The main charater's feeling " + '"' +str(topicAkeyword) + '"' +  " is most likely related to the statement: "
        print('Stem: ', stem)
        
        keyed_option = sent_score(text, topicA)
        distractors = []
        for i in range(1, 11):
            distractors.append(sent_score(text, i))
        distractors= [sub for lst in distractors for sub in lst]
        distractors = [i for i in list(set(distractors)) if i not in keyed_option]
        q.append((text, stem, keyed_option, distractors))
        
    return q
    
  
def item_model2(data = coherent, df_wrd=df_wrd, num= i):
    
    text = data.samples.iloc[num]
    
    try:
        text = clean_text(text)
    except:
        text = text
    topic_index =data.drop(columns=['chapter', 'total', 'samples', 'dominant','length']).iloc[num]
    topicA = topic_index.argmax()+1
    print('---- Topic A is Topic: ', topicA)
    keysent_list = sent_score(text, topicA)
    
    
    stem= "What can be reasonably inferred from the " + '"' +str(keysent_list) + '"' +  " of the passage that that the main character felt "
    print('Stem: ', stem)
    
    keyed_option = df_wrd.iloc[topicA-1].sort_values()[-10:].index
    
    distractors = []
    for i in range(1, 11):
        distractors.append(df_wrd.iloc[i-1].sort_values()[-10:].index)
    distractors= [sub for lst in distractors for sub in lst]
    distractors = [i for i in list(set(distractors)) if i not in keyed_option]
    
    return (text, stem, keyed_option, distractors)
    
    
def item_model3(data = divergent, df_wrd=df_wrd, num= i):
    
    text = data.samples.iloc[num]
    try:
        text = clean_text(text)
    except:
        text = text
    topic_index =data.drop(columns=['chapter', 'total', 'samples', 'dominant','length']).iloc[num]
    topicA = topic_index.argmax()+1
    topicB = int(topic_index.sort_values().index[-2].split('. ')[-1])
    
    print('---- Topic A is Topic: ', topicA)
    print('---- Topic B is Topic: ', topicB)
    keysentA = sent_score(text, topicA)
    keysentB = sent_score(text, topicB)
    
    stem= "How is the main characters sentiment described in " +str(keysentA) + " and " + str(keysentB) + " different?"
    print('Stem: ', stem)

    keyed_optionA =df_wrd.iloc[topicA-1].sort_values()[-20:].index
    keyed_optionB =df_wrd.iloc[topicB-1].sort_values()[-20:].index
    
    # other_topics = [i for i in topic_index if ]
    #keyed_option = itertools.combinations(keyed_optionA,keyed_optionB)
    distractors = []
    #orrect-incorrect = intertools.
    
    return (text, stem, keyed_optionA, keyed_optionB, distractors)  
    
  
def item_model4(data = divergent, df_wrd=df_wrd, num= i):
    
    text = data.samples.iloc[num]
    text = clean_text(text)
    topic_index =data.drop(columns=['chapter', 'total', 'samples', 'dominant','length']).iloc[num]
    
    q = []
    for k in topic_index[topic_index!=0].index:
        topicA = int(k.split('. ')[-1])
        
    
        stem= "Which of the following indicates different sentiment from the others?"
        print('Stem: ', stem)
    
        keyed_optionA = sent_score(text, topicA)
        distractors = [sent_score(text, i) for i in range(1,11) if i != topicA]
        distractors= list(set([sub for lst in distractors for sub in lst]))
        distractors= [i for i in distractors if i not in keyed_optionA]
        
        q1 = [text, stem, keyed_optionA, distractors]
        print('---- Topic A is Topic: ', topicA)
        print('---- Topic B is Topic: ', topicB)
        
        stem= "Which of the following indicates different sentiment from the others?"
        print('Stem: ', stem)

        keyed_optionA = sent_score(text, topicA)
        distractors = [sent_score(text, i) for i in range(1,11) if i != topicA]
        distractors= list(set([sub for lst in distractors for sub in lst]))
        distractors= [i for i in distractors if i not in keyed_optionA]
        q.append((text, stem, keyed_optionA, distractors))
    return q
#%%
c1q = []
for i in range(len(coherent)):
    c1q.append(item_model1(coherent, df_wrd, i))

c1q = [sub for lst in c1q for sub in lst] #34080 items  
c1q_df = pd.DataFrame(c1q)
#c1q_df.drop_duplicates(keep='first')
c1q_df['n_key'] = c1q_df[2].apply(len)
c1q_df['n_distractor'] = c1q_df[3].apply(len)

c1q_df = c1q_df[c1q_df.n_distractor >1]
#%%


with open('coherent_item_model_1.txt', 'w') as t:
    p=1
    for i in c1q_flat:
        
        t.write('Q'+str(p)+'.'+'\n')
        t.write(i[0]+'\n\n')
        t.write(i[1]+'\n\n')
        t.write('Answer' +'\n')
        t.write(str(i[2])+'\n')
        t.write('Distractors' +'\n')
        t.write(str(i[3]))
        t.write('\n\n')
        t.write('--------------------------------------------------------------')
        t.write('\n\n')
        p+=1
        
#%%
c2q = []
for i in range(len(coherent)):
    c2q.append(item_model2(coherent, df_wrd, i))
    
c2q_df = pd.DataFrame(c2q)
#c1q_df.drop_duplicates(keep='first')
c2q_df['n_key'] = c2q_df[2].apply(len)
c2q_df['n_distractor'] = c2q_df[3].apply(len)

c2q_df = c2q_df[c2q_df.n_distractor >1]
#%%
#c1q_flat = [sub for lst in c1q for sub in lst] #34080 items 

with open('coherent_item_model_2.txt', 'w') as t:
    p=1
    for i in c2q:
        
        t.write('Q'+str(p)+'.'+'\n')
        t.write(i[0]+'\n\n')
        t.write(i[1]+'\n\n')
        t.write('Answer' +'\n')
        t.write(str(i[2])+'\n')
        t.write('Distractors' +'\n')
        t.write(str(i[3]))
        t.write('\n\n')
        t.write('--------------------------------------------------------------')
        t.write('\n\n')
        p+=1
        
#%%
c3q = []
for i in range(len(divergent)):
    c3q.append(item_model3(divergent, df_wrd, i))
    
c3q_df = pd.DataFrame(c3q)
#c1q_df.drop_duplicates(keep='first')
c3q_df['n_key'] = c3q_df[2].apply(len)
c3q_df['n_distractor'] = c3q_df[3].apply(len)

c3q_df = c3q_df[c3q_df.n_distractor >1]
#%%

with open('divergent_item_model_1.txt', 'w') as t:
    p=1
    for i in c3q:
        
        t.write('Q'+str(p)+'.'+'\n')
        t.write(i[0]+'\n\n')
        t.write(i[1]+'\n\n')
        t.write('Answer' +'\n')
        t.write(str(i[2])+'\n')
        t.write('Distractors' +'\n')
        t.write(str(i[3]))
        t.write('\n\n')
        t.write('--------------------------------------------------------------')
        t.write('\n\n')
        p+=1
        
#%%
c4q = []
for i in range(len(divergent)):
    c4q.append(item_model4(divergent, df_wrd, i))

c4q = [sub for lst in c4q for sub in lst] #34080 items  
c4q_df = pd.DataFrame(c4q)
#c1q_df.drop_duplicates(keep='first')
c4q_df['n_key'] = c4q_df[2].apply(len)
c4q_df['n_distractor'] = c4q_df[3].apply(len)

c4q_df = c4q_df[c4q_df.n_distractor >1]
#%%


with open('divergent_item_model_2.txt', 'w') as t:
    p=1
    for i in c4q:
        
        t.write('Q'+str(p)+'.'+'\n')
        t.write(i[0]+'\n\n')
        t.write(i[1]+'\n\n')
        t.write('Answer' +'\n')
        t.write(str(i[2])+'\n')
        t.write('Distractors' +'\n')
        t.write(str(i[3]))
        t.write('\n\n')
        t.write('--------------------------------------------------------------')
        t.write('\n\n')
        p+=1
         
    
    
    