
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data=pd.DataFrame({
    'id':[1,2,3,4,5],
    'description':['Virat Kohli is a good cricketer and a sport person, ho plays
cricket well',
                   'cricket is a famous sports in India and people likes to play
it',
                   'artificial intelligence is the subject taught in semester 6
of department of information and communication technnology',
                   'natural language processing and recommendation systems are
the topics of artificial intelligence',
                   'the world is currently blessed with the fruits of artificial
intelligence']
})
#word embedding vectors
tfidf=TfidfVectorizer(stop_words='english')
tfidf_matrix=tfidf.fit_transform(data['description'])

tfidf_matrix

recommendation_of_item=3

cosine=cosine_similarity(tfidf_matrix,tfidf_matrix)

sim_scores=list(enumerate(cosine[recommendation_of_item]))

sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)

print(sim_scores)

top_n=2

extracted_items=sim_scores[1:top_n+1]
indices=[i[0] for i in extracted_items]
print(data.iloc[indices])
