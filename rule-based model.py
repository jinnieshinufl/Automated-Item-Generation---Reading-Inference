
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:48:54 2021

@author: jinnieshin
"""

# rule-based parsing system  (subtext parsing)

# we will check the density by entropy 
                                   
candidates = df_document_topic #pd.read_excel('subtext_candidates.xlsx')
candidates['samples'] = files.text.tolist()
# one sample example: 
#candidates.drop(columns=['chapter', 'total','samples']).loc[0].plot.bar()
#candidates = candidates.drop(columns=['Dominant topic']) #removing all the NAs 

#if it is more than 90% about one topic 
#%%
dominant = ((candidates.drop(columns=['Dominant topic', 'samples']) >= 0.7)*1).sum(axis=1)
candidates['dominant'] = dominant
# 1,006 topics are dominant in their one topic structure 
#%%

import matplotlib.pyplot as plt
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
plt.figure(figsize=(16, 6))

import itertools
#computing sematic similarities between the topics
'''Kullback–Leibler divergence (KLD) between distribution of 
words in two topics KLD(T1|T2) shows how much information (in bits) 
is lost when T2 is being used to model T1 data. it is a non negative 
measure that my have correlation with semantic similarity. 
KLD is not symmetric, KLD(T1|T2) not equal to  KLD(T2|T1) but 
it can be simply symmetrize by arithmetic or geometric average 
averaging of KLD(T1|T2) and KLD(T2|T1)'''

df_wrd = df.T
df_wrd = df_wrd.div(df_wrd.sum(axis=1), axis=0)

from scipy.spatial.distance import jensenshannon
from numpy import asarray
# define distributions

# calculate JS(P || Q)
dic=dict()
for i in list(itertools.combinations(df_wrd.T.columns,2)):
    js_pq = jensenshannon(df_wrd.T[i[0]].values, df_wrd.T[i[1]].values, base=2)
    print('JS(P || Q) Distance: %.3f' % js_pq)
    dic[i] = js_pq # all of the things were maximumly different 
    
# all the samples seemed to be relatively distinct - divergent 
#%%
coherent = candidates[candidates.dominant==1]
divergent = candidates[candidates.dominant==0]
#%%


