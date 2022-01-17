import pandas as pd
import numpy as np
import gensim.downloader as api
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity
import gensim

vectors = gensim.models.KeyedVectors.load_word2vec_format('./models/ubercorpus.lowercased.lemmatized.word2vec.300d')
ukr_words = ['машина', 'колесо', 'автомобіль', 'літак', 'небо',"вовк", "лиса", "заєць", "казка", "легенда", "байка", "оповідання", "повість", "балада", "інформатика", "математика", "індустрія", "дельтаплан"]

# ukr_words = [
#     "предмет",
#     "предмет",
#     "інформатика",
#     "розвиток",
#     "індустрія",
#     "протяг",
#     "десятка",
#     "рік",
#     "процес",
#     "самовизначення",
#     "самовизначення",
#     "інформатика",
#     "наука",
#     "боротьба",
#     "полюс",
#     "підґрунтя",
#     "аспект",
#     "процес",
#     "обробка",
#     "процес",
#     "обробка",
#     "інформація",
#     "обробка",
#     "інформація",
#     "так",
#     "підхід",
#     "інженерний",
#     "підхід",
#     "система",
#     "період",
#     "становлення",
#     "становлення",
#     "інформатика",
#     "коли",
#     "комп'ютерний",
#     "система",
#     "як",
#     "приклад",
#     "думка",
#     "предмет",
#     "інформатика",
#     "вивчення",
#     "інформатика",
#     "вивчення",
#     "вивчення",
#     "машина",
#     "комплекс",
#     "проблема",
#     "проектування",
#     "розробка",
#     "під",
#     "відомий",
#     "під",
#     "назва",
#     "загальний",
#     "назва",
#     "інженерія",
#     "план",
#     "проблема",
#     "взаємодія",
#     "суб'єкт",
#     "середина",
#     "комунікативний",
#     "процес",
#     "математичний",
#     "підхід",
#     "проникнення",
#     "тотальний",
#     "проникнення",
#     "технологія",
#     "сфера",
#     "сфера",
#     "життя",
#     "сфера",
#     "життя",
#     "суспільство",
#     "життя",
#     "життя",
#     "суспільство",
#     "підвищення",
#     "підвищення",
#     "різкий",
#     "підвищення",
#     "підвищення",
#     "проблема",
#     "підвищення",
#     "проблема",
#     "підвищення",
#     "продуктивність",
#     "праця",
#     "галузь",
#     "продукування",
#     "підвищення",
#     "надійність",
#     "технологія",
#     "надійність",
#     "зниження",
#     "суттєвий",
#     "зниження",
#     "вартість",
#     "мова",
#     "запровадження",
#     "метод",
#     "виробництво",
#     "комп'ютер",
#     "забезпечення",
#     "програмний",
#     "забезпечення",
#     "подібний",
#     "виробництво",
#     "автоматизація",
#     "підтримка",
#     "математичний",
#     "підтримка",
#     "ряд",
#     "цілий",
#     "ряд",
#     "розділ",
#     "математика",
#     "математика",
#     "логіка",
#     "логіка",
#     "семіотика",
#     "лінгвістика",
#     "документація",
#     "експлуатація",
#     "експлуатація",
#     "система",
#     "підстава",
#     "дата",
#     "вовк",
#     "лиса",
#     "заєць",
#     "казка",
#     "легенда",
#     "байка",
#     "оповідання",
#     "повість",
#     "балада",
#     "мультфільм",
#     "коза-дереза",
#     "колобок",
#     "поема",
#     "вірш",
#     "літак",
#     "вертоліт",
#     "гелікоптер",
#     "планер",
#     "аеробус",
#     "як-40",
#     "бомбардувальник",
#     "авіатехнік",
#     "дельтаплан",
# ]

words_vectors = np.array([vectors[x] for x in ukr_words])
vectors = minmax_scale(words_vectors, feature_range=(0, 1))

def get_sim_matrix(X, threshold=0.99):
    """Output pairwise cosine similarities in X. Cancel out the bottom
    `threshold` percent of the pairs sorted by similarity.
    """
    sim_matrix = cosine_similarity(X)
    np.fill_diagonal(sim_matrix, 0.0)

    sorted_sims = np.sort(sim_matrix.flatten())
    thr_val = sorted_sims[int(sorted_sims.shape[0]*threshold)]
    sim_matrix[sim_matrix < thr_val] = 0.0

    return sim_matrix

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

sim_matrix = get_sim_matrix(vectors, 0.8)
labels = majorclust(sim_matrix)
word2clusters = dict(zip(ukr_words, labels))

print(word2clusters)