import nltk
from ether.instance.pool.filter import nltkFilter
import numpy as np

datax = 'I see a bird flying on top of the building'
datax = '%s %s %s' % (nltkFilter.sentence_start_token, datax, nltkFilter.sentence_end_token)
datax = nltk.word_tokenize(datax)
word_freq = nltk.FreqDist(datax)
vocab = word_freq.most_common(nltkFilter.vocabulary_size-1)

'''
re-order the words, restrict their index according to their frequency
'''
index_to_word = [x[0] for x in vocab]
index_to_word.append(nltkFilter.unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

# Replace all words not in our vocabulary with the unknown token
datax = [w if w in word_to_index else nltkFilter.unknown_token for w in datax]

y_train = np.asarray([word_to_index[w] for w in datax[1:]])
