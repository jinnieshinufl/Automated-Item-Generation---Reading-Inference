# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 22:36:01 2022

@author: Jinnie Shin 
"""
import nltk
import re
import os
from nltk import sent_tokenize, word_tokenize
from collections import defaultdict
from nltk.corpus import stopwords
from pattern.en import parse, Sentence, mood
from glob import glob
import pandas as pd 

def clean_str(text, remove_stopwords=True, stem_words=True):
    # Clean the text, with the option to remove stopwords and to stem words.
    # Convert words to lower case and split them
    text = text.split()
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z(),!?]", " ", str(text))
    text = re.sub(r"\, said", "said", str(text))
    text = re.sub(r"\'s", " \'s", str(text))
    text = re.sub(r"\'ve", " \'ve", str(text))
    text = re.sub(r"n\'t", " n\'t", str(text))
    text = re.sub(r"\'re", " \'re", str(text))
    text = re.sub(r"\'d", " \'d", str(text))
    text = re.sub(r"\'ll", " \'ll", str(text))
    text = re.sub(r",", " , ", str(text))
    text = re.sub(r"!", " ! ", str(text))
    text = re.sub(r"\?", " \? ", str(text))
    text = re.sub(r"\s{2,}", " ", str(text))
    return(text)
    
def chunkSentences(text):

    sentences = nltk.sent_tokenize(text)
    tokenizedSentences = [nltk.word_tokenize(sentence)
                          for sentence in sentences]
    taggedSentences = [nltk.pos_tag(sentence)
                       for sentence in tokenizedSentences]
    if nltk.__version__[0:2] == "2.":
        chunkedSentences = nltk.batch_ne_chunk(taggedSentences, binary=True)
    else:
        chunkedSentences = nltk.ne_chunk_sents(taggedSentences, binary=True)
    return chunkedSentences


def extractEntityNames(tree, _entityNames=None):
    if _entityNames is None:
        _entityNames = []
    try:
        if nltk.__version__[0:2] == "2.":
            label = tree.node
        else:
            label = tree.label()
    except AttributeError:
        pass
    else:
        if label == 'NE':
            _entityNames.append(' '.join([child[0] for child in tree]))
        else:
            for child in tree:
                extractEntityNames(child, _entityNames=_entityNames)
    return _entityNames


def buildDict(chunkedSentences, _entityNames=None):
    if _entityNames is None:
        _entityNames = []

    for tree in chunkedSentences:
        extractEntityNames(tree, _entityNames=_entityNames)

    return _entityNames


def removeStopwords(entityNames, customStopWords=None):
    # Memoize custom stop words
    if customStopWords is None:
        customStopWords = ['AND','THE','Street', 'Road', 'Bridge', 'Town', 'Park', 'Hill', 'Lane', 'CHAPTER', 'Chapter', 'CHAPTER I', 'CHAPTER II', 'CHAPTER III', 'CHAPTER IV', 
                          'BOOK', 'Book', 'Great', 'Are', 'Guard', 'Wait', 'Him', 'Her', 'Look', 'Everything', 'Toward', 'Thy', 'Everyone', 'Every', 'Project', 'Project Gutenberg']

    for name in entityNames:
        if name in stopwords.words('english') or name in customStopWords:
            entityNames.remove(name)


def getMajorCharacters(entityNames):
    return list({name for name in entityNames if entityNames.count(name) > 10})


def splitIntoSentences(text):
    
    sentenceEnders = re.compile(r"""
    # Split sentences on whitespace between them.
    (?:               # Group for two positive lookbehinds.
      (?<=[.!?])      # Either an end of sentence punct,
    | (?<=[.!?]['"])  # or end of sentence punct and quote.
    )                 # End group of two positive lookbehinds.
    (?<!  Mr\.   )    # Don't end sentence on "Mr."
    (?<!  Mrs\.  )    # Don't end sentence on "Mrs."
    (?<!  Ms\.   )    # Don't end sentence on "Ms."
    (?<!  Jr\.   )    # Don't end sentence on "Jr."
    (?<!  Dr\.   )    # Don't end sentence on "Dr."
    (?<!  Prof\. )    # Don't end sentence on "Prof."
    (?<!  Sr\.   )    # Don't end sentence on "Sr."
    \s+               # Split on whitespace between sentences.
    """, re.IGNORECASE | re.VERBOSE)
    
    return sentenceEnders.split(text)


def compareLists(sentenceList, majorCharacters):
    characterSentences = defaultdict(list)
    for sentence in sentenceList:
        for name in majorCharacters:
            if re.search(r"\b(?=\w)%s\b(?!\w)" % re.escape(name),
                         sentence,
                         re.IGNORECASE):
                characterSentences[name].append(sentence)
    return characterSentences


def extractMood(characterSentences):
    characterMoods = defaultdict(list)
    for key, value in characterSentences.items():
        for x in value:
            characterMoods[key].append(mood(Sentence(parse(str(x),
                                                           lemmata=True))))
    
    return characterMoods

def CharacterAnalysis(text):
    Cast =[]
    Characters= {}    
    CharacterMoods ={}
    CharacterTones = {}
    for i in range(len(text)):
        chunkedSentences = chunkSentences(text[i])
        entityNames = buildDict(chunkedSentences)
        removeStopwords(entityNames)
        majorCharacters = getMajorCharacters(entityNames)
        Characters['Chapter '+str(i+1)] = majorCharacters
        Cast.append(majorCharacters)
        
        sentenceList = splitIntoSentences(text[i])
        characterSentences = compareLists(sentenceList, majorCharacters)
        characterMoods = extractMood(characterSentences)
        CharacterMoods['Chapter '+str(i+1)] = characterMoods
    
    
    return Cast, Characters, CharacterMoods, CharacterTones 

def dialogue(sent):
    if '"' in sent:
        return True
    else:
        return False

def flatten(l):
    flat_list = [item for sublist in l for item in sublist]
    return flat_list


def get_text_data():

    files = sorted(glob(os.path.join("./data/Chapter*.txt")),key=lambda chapter: int(re.search(r'\d+',chapter).group()))
    if len(files) == 0:
        files = sorted(glob(os.path.join(".\data\Chapter*.txt")),key=lambda chapter: int(re.search(r'\d+',chapter).group()))

    text = []
    for file in files:
        with open(file, 'r',encoding='utf8') as f:
            temp=[]
            for lines in f:
                lines =lines.replace('\n', '')
                lines = lines.replace('“','"')
                lines = lines.replace('”', '"')
                temp.append(lines)
            temp = ','.join(temp)
            #temp = clean_str(temp)
            text.append(temp)#%%    
    chars = pd.read_excel('entities.xlsx')[0].tolist()
    
    def charsearch2(text, chars):
        sent = sent_tokenize(text)
        output = []
        for i in range(len(sent)):
            if any([x in sent[i] for x in chars]):
                output.append(sent[i])
            elif dialogue(sent[i]) == True:
                output.append(sent[i])
            else:
                output = output
        return output        
        

    Relate= []
    for k in range(len(text)):
        temp = charsearch2(text[k], chars)
        Relate.append(temp)
    
    input_text = []
    for i in range(len(Relate)):
        temp = (' ').join(Relate[i])
        input_text.append(temp)
    
    return input_text
    