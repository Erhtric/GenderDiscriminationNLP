import os
from tkinter import W
import numpy as np
import pickle
from collections import Counter, defaultdict
from tqdm import tqdm
from data import load_text8

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

def DEFUALT_IDX(): return 0
def DEFAULT_WORD(): return '<UNK>'
def DEFAULT_GENDER(): return 0

def get_vocab_data(corpus, target_vocab_size=100000):
  """
  Returns the mapping word to index (and viceversa) of the first [target_vocab_size] of the [corpus].
  """
  token_counter = Counter(corpus)

  # we can further remove the get_idx method by using instead of a dict a Counter
  word_to_idx = defaultdict(DEFUALT_IDX)
  word_to_idx['<UNK>'] = 0
  idx_to_word = defaultdict(DEFAULT_WORD)
  idx_to_word[0] = '<UNK>'

  for idx, (key, _) in enumerate(tqdm(token_counter.most_common(target_vocab_size-1)), 1):
    word_to_idx[key] = idx
    idx_to_word[idx] = key

  return word_to_idx, idx_to_word

def get_gender(id, idx_to_word, gendered_words):
  return gendered_words[idx_to_word[id]]

def get_gender_file():
  gendered_words = defaultdict(DEFAULT_GENDER)
  with open(os.path.join(PKG_DIR, '../data/english_data', "gendered_words.txt"), "r") as f:
    lines = f.readlines()
    for line in lines:
      word, gender = line.strip().split(" ")
      if gender == "n":
        gender = 0
      elif gender =="m":
        gender = 1
      else:
        gender = 2
      gendered_words[word] = gender
  return gendered_words

def get_en_gender_sets():
  """
  Load the english gender sets from the data folder
  """
  with open(os.path.join(PKG_DIR, '../data/english_data', "male_word_file.txt"), "r") as f :
    male_words = list(set(map(lambda x: x.strip(), (f.readlines()))))

  with open(os.path.join(PKG_DIR, '../data/english_data', "female_word_file.txt"), "r") as f :
    female_words = list(set(map(lambda x: x.strip(), (f.readlines()))))

  return male_words, female_words

def compute_gender_indexes(male_set, female_set, word_to_idx):
  """Compute the gender indexes given [word_to_idx].

  Args:
      male_set (_type_): _description_
      female_set (_type_): _description_
      word_to_idx (_type_): _description_

  Returns:
      _type_: _description_
  """
  male_idxs = np.array(list(filter(lambda n: n != 0, map(lambda x: word_to_idx[x], male_set))))
  female_idxs = np.array(list(filter(lambda n: n != 0, map(lambda x: word_to_idx[x], female_set))))
  print(f'Number of male words: {male_idxs.shape[0]}, number of female words: {female_idxs.shape[0]}')
  return male_idxs, female_idxs

def partitioned_generate_cooccurrence_matrix(corpus,
                                            window_size,
                                            word_to_idx,
                                            idx_to_word,
                                            gendered_words,
                                            partition_size):
  co_occurrences_counter = Counter()

  for i in tqdm(range(0, len(corpus), partition_size)):
    partition = corpus[i:i+partition_size]
    size = len(partition)

    for i, word in enumerate(partition):
      end = min(len(partition), i + window_size + 1)
      start = max(0, i - window_size)

      for j, context_word in enumerate(partition[start:end]):
        if i != j:
          co_occurrences_counter[
            word_to_idx[word],
            word_to_idx[context_word],
          ] += 1 / abs(i - j)

  # row, col, value, gender
  co_matrix = np.zeros((len(co_occurrences_counter), 5))
  for n, ((id_word, id_context), value) in enumerate(co_occurrences_counter.items()):
    gender_target = get_gender(id_word, idx_to_word, gendered_words)
    gender_context = get_gender(id_context, idx_to_word, gendered_words)
    co_matrix[n] = np.array([id_word, id_context, value, gender_target, gender_context])

  return co_matrix

def generate_cooccurrence_matrix(corpus, window_size, num_partitions, gendered_words):
  word_to_idx, idx_to_word = get_vocab_data(corpus)
  partition_size = len(corpus)//num_partitions

  print(f'Length of the corpus: {len(corpus)}, number of partitions selected: {num_partitions}')
  print(f'Partitions\' size (words): {partition_size}')

  co_occurrence_matrix = partitioned_generate_cooccurrence_matrix(corpus,
                                                                  window_size,
                                                                  word_to_idx,
                                                                  idx_to_word,
                                                                  gendered_words,
                                                                  partition_size)

  # Save to file
  path = os.path.join(PKG_DIR, '../data/english_data', f"cooccurrence_matrix_copy_w{window_size}.pkl")
  with open(path, "wb+") as f:
    pickle.dump(co_occurrence_matrix, f)

  return co_occurrence_matrix

def load_cooccurrence_matrix(window_size=4, partitions=1700):
  window_size = 5
  matrixpath = os.path.join(PKG_DIR, '../data/english_data', f"cooccurrence_matrix_copy_w{window_size}.pkl")
  if not os.path.isfile(matrixpath):
    corpus = load_text8()
    gendered_words = get_gender_file()
    co_matrix_gender = generate_cooccurrence_matrix(corpus, window_size, partitions, gendered_words)
    return co_matrix_gender.astype(np.float32)
  else:
    with open(matrixpath, 'rb') as f:
      co_matrix_gender = pickle.load(f)
      return co_matrix_gender.astype(np.float32)

if __name__ == "__main__":
    corpus = load_text8()
    # word_to_idx, idx_to_word = get_vocab_data(corpus)
    # male_words, female_words = get_en_gender_sets()
    # male_idxs, female_idxs = compute_gender_indexes(male_words, female_words, word_to_idx)
    
    co_matrix_gender = load_cooccurrence_matrix()