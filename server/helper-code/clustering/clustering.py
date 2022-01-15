import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity
import gensim

ubercorpus_model = gensim.models.KeyedVectors.load_word2vec_format('./models/ubercorpus.lowercased.lemmatized.word2vec.300d')
ubercorpus_vectors = ubercorpus_model
# ukr_words = ['машина', 'колесо', 'автомобіль', 'літак', 'небо']
ukr_words = [
    "предмет",
    "предмет",
    "інформатика",
    "розвиток",
    "індустрія",
    "протяг",
    "десятка",
    "рік",
    "процес",
    "самовизначення",
    "самовизначення",
    "інформатика",
    "наука",
    "боротьба",
    "полюс",
    "підґрунтя",
    "аспект",
    "процес",
    "обробка",
    "процес",
    "обробка",
    "інформація",
    "обробка",
    "інформація",
    "так",
    "підхід",
    "інженерний",
    "підхід",
    "система",
    "період",
    "становлення",
    "становлення",
    "інформатика",
    "коли",
    "комп'ютерний",
    "система",
    "як",
    "приклад",
    "думка",
    "предмет",
    "інформатика",
    "вивчення",
    "інформатика",
    "вивчення",
    "вивчення",
    "машина",
    "комплекс",
    "проблема",
    "проектування",
    "розробка",
    "під",
    "відомий",
    "під",
    "назва",
    "загальний",
    "назва",
    "інженерія",
    "план",
    "проблема",
    "взаємодія",
    "суб'єкт",
    "середина",
    "комунікативний",
    "процес",
    "математичний",
    "підхід",
    "проникнення",
    "тотальний",
    "проникнення",
    "технологія",
    "сфера",
    "сфера",
    "життя",
    "сфера",
    "життя",
    "суспільство",
    "життя",
    "життя",
    "суспільство",
    "підвищення",
    "підвищення",
    "різкий",
    "підвищення",
    "підвищення",
    "проблема",
    "підвищення",
    "проблема",
    "підвищення",
    "продуктивність",
    "праця",
    "галузь",
    "продукування",
    "підвищення",
    "надійність",
    "технологія",
    "надійність",
    "зниження",
    "суттєвий",
    "зниження",
    "вартість",
    "мова",
    "запровадження",
    "метод",
    "виробництво",
    "комп'ютер",
    "забезпечення",
    "програмний",
    "забезпечення",
    "подібний",
    "виробництво",
    "автоматизація",
    "підтримка",
    "математичний",
    "підтримка",
    "ряд",
    "цілий",
    "ряд",
    "розділ",
    "математика",
    "математика",
    "логіка",
    "логіка",
    "семіотика",
    "лінгвістика",
    "документація",
    "експлуатація",
    "експлуатація",
    "система",
    "підстава",
    "дата"
]

ukr_words_vectors = np.array([ubercorpus_vectors[x] for x in ukr_words])
ubercorpus_vectors = minmax_scale(ukr_words_vectors, feature_range=(0, 1))

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
ukr_sim_matrix = get_sim_matrix(ubercorpus_vectors, 0.8)

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

ukr_labels = majorclust(ukr_sim_matrix)
ukr_word2clusters = dict(zip(ukr_words, ukr_labels))

print(ukr_word2clusters)