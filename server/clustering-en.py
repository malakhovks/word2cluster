import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity


vectors = api.load("glove-wiki-gigaword-100")
words = ['car', 'vehicle', 'bike', 'helmet', 'cycle', 'motorcycle', 'pickup', 'game', 'xbox', 'player', 'warfare', 'battlefield', 'playstation', 'wii', 'nintendo', 'fifa', 'controller', 'sims', 'console', 'ford', 'toyota', 'bmw', 'audi', 'honda', 'tesla', 'nissan', 'jeep', 'kia', 'suv', 'ferrari']

word_vectors = np.array([vectors[x] for x in words])

vectors = minmax_scale(word_vectors, feature_range=(0, 1))

def get_sim_matrix(X, threshold=0.9):
    """Output pairwise cosine similarities in X. Cancel out the bottom
    `threshold` percent of the pairs sorted by similarity.
    """
    sim_matrix = cosine_similarity(X)
    np.fill_diagonal(sim_matrix, 0.0)

    sorted_sims = np.sort(sim_matrix.flatten())
    thr_val = sorted_sims[int(sorted_sims.shape[0]*threshold)]
    sim_matrix[sim_matrix < thr_val] = 0.0

    return sim_matrix

sim_matrix = get_sim_matrix(vectors, 0.8)

def majorclust(sim_matrix):
    """Actual MajorClust algorithm
    """
    t = False
    indices = np.arange(sim_matrix.shape[0])
    while not t:
        t = True
        for index in np.arange(sim_matrix.shape[0]):
            # check if all the sims of the word are not zeros
            weights = sim_matrix[index]
            if weights[weights > 0.0].shape[0] == 0:
                continue
            # aggregating edge weights
            new_index = np.argmax(np.bincount(indices, weights=weights))
            if indices[new_index] != indices[index]:
                indices[index] = indices[new_index]
                t = False
    return indices

labels = majorclust(sim_matrix)
word2clusters = dict(zip(words, labels))

print(word2clusters)