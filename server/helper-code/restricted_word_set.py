import numpy as np
import gensim
import random

# def restrict_w2v(w2v, restricted_word_set):
#     new_vectors = []
#     new_vocab = {}
#     new_index2entity = []
#     new_vectors_norm = []

#     for i in range(len(w2v.vocab)):
#         word = w2v.index2entity[i]
#         vec = w2v.vectors[i]
#         vocab = w2v.vocab[word]
#         vec_norm = w2v.vectors_norm[i]
#         if word in restricted_word_set:
#             vocab.index = len(new_index2entity)
#             new_index2entity.append(word)
#             new_vocab[word] = vocab
#             new_vectors.append(vec)
#             new_vectors_norm.append(vec_norm)

#     w2v.vocab = new_vocab
#     w2v.vectors = np.array(new_vectors)
#     w2v.index2entity = np.array(new_index2entity)
#     w2v.index2word = np.array(new_index2entity)
#     w2v.vectors_norm = np.array(new_vectors_norm)

# w2v = gensim.models.KeyedVectors.load_word2vec_format("../models/ubercorpus.lowercased.lemmatized.word2vec.c.text.format.300d")
# print(w2v.most_similar("казка"))

# restricted_word_set = {"казка", "байка", "монітор", "планшет", "балада", "вірш"}
# restrict_w2v(w2v, restricted_word_set)
# print(w2v.most_similar("казка"))

w2v = gensim.models.KeyedVectors.load_word2vec_format("../models/ubercorpus.lowercased.lemmatized.word2vec.c.text.format.300d")
print(w2v.most_similar_to_given("балада", ["байка", "монітор", "планшет", "балада", "вірш"]))
print(w2v.distances("балада", ["байка", "монітор", "планшет", "балада", "вірш"]))
# random_word = random.choice(w2v.index_to_key)
random_word = random.sample(w2v.index_to_key, 10)
print("Random word: "+ str(random_word))