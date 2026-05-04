
#textRank for Document Summarisation

#import basic libraries
import nltk
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

#Downloading nltk resources

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

def preprocessing(text):
  stop_words=set(stopwords.words('english'))
  sentences=sent_tokenize(text)
  word_frequencies=[]
  for sent in sentences:
    words=word_tokenize(sent)
    filtered_words=[word for word in words if word.isalnum()
                      and word not in stop_words]
    word_frequencies.append(Counter(filtered_words))
  return sentences,word_frequencies

def similarity_matrix(word_frequencies):
  size=len(word_frequencies)
  sm=np.zeros((size,size))
  for i in range(size):
    for j in range(size):
      if i!=j:
        words1=word_frequencies[i]
        words2=word_frequencies[j]
        common_words=set(words1.keys()).union(set(words2.keys()))

        vec1=np.array([words1[word] for word in common_words])
        vec2=np.array([words2[word] for word in common_words])

        sm[i][j]=cosine_similarity([vec1],[vec2])[0,0]
  return sm

def textrank_summarization(text,num_sentences=3):
  sentences,word_frequencies=preprocessing(text)
  sm=similarity_matrix(word_frequencies)
  #building graph
  #print(sm)
  graph=nx.from_numpy_array(sm)
  scores=nx.pagerank(graph)
  #print(scores)
  ranked_sentences = sorted(((scores.get(i, 0), s) for i, s in enumerate(sentences)),
    reverse=True)
  summary = " ".join([sent for _, sent in ranked_sentences[:num_sentences]])

  return summary

text="""The UEFA Champions League (UCL or UEFA CL), commonly known as the Champions League, is an annual club association football competition organised by the Union of European Football Associations (UEFA) that is contested by top-division European clubs. The competition begins with a round robin league phase to qualify for the double-legged knockout rounds, and a single-leg final. It is the most-watched club competition in the world and the third most-watched football competition overall, behind only the FIFA World Cup and the UEFA European Championship. It is one of the most prestigious football tournaments in the world and the most prestigious club competition in European football, played by the national league champions (and, for some nations, one or more runners-up) of their national associations.

Introduced in 1955 as the European Champion Clubs' Cup (French: Coupe des Clubs Champions Européens), and commonly known as the European Cup, it was initially a straight knockout tournament open only to the champions of Europe's domestic leagues, with its winner reckoned as the European club champion. The competition took on its current name in 1992, adding a round-robin group stage in 1991 and allowing multiple entrants from certain countries since the 1997–98 season.[1] While only the winners of many of Europe's national leagues can enter the competition, the top 5 leagues by coefficient provide four teams each by default,[2] with a possibility for additional spots based on performance during the previous season.[3][4] Clubs that finish below the qualifying spots are eligible for the second-tier UEFA Europa League competition, and since 2021, for the third-tier UEFA Conference League"""

textrank_summarization(text)
