import os
import numpy as np
import pickle
from collections import Counter
from tqdm import tqdm
from data import load_text8

PKG_DIR = os.path.dirname(os.path.abspath(__file__))

def get_vocab_data(corpus, target_vocab_size=100000):
  """
  Returns the mapping word to index (and viceversa) of the first [target_vocab_size] of the [corpus].
  """
  token_counter = Counter(corpus)

  # we can further remove the get_idx method by using instead of a dict a Counter
  word_to_idx = dict()
  word_to_idx['<UNK>'] = 0
  idx_to_word = dict()
  idx_to_word[0] = '<UNK>'

  for idx, (key, _) in enumerate(tqdm(token_counter.most_common(target_vocab_size-1)), 1):
    word_to_idx[key] = idx
    idx_to_word[idx] = key

  return word_to_idx, idx_to_word

def get_idx(token, word_to_idx):
  """
  Checks if a word is in the vocabulary, if so then it returns the associated word, otherwise
  return the value associated to the unknown word.
  """
  if token in word_to_idx:
    return word_to_idx[token]
  return word_to_idx['<UNK>']

def get_gender(target, context, male_idxs, female_idxs):
  """Label the target with the appropriate gender.

  Args:
      target (_type_): _description_
      context (_type_): _description_
      male_idxs (_type_): _description_
      female_idxs (_type_): _description_

  Returns:
      int: the label with the relative gender
  """
  if target in male_idxs:
    return 0
  if context in male_idxs:
    return 1
  if target in female_idxs:
    return 2
  if context in female_idxs:
    return 3
  return 4

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
  male_idxs = np.array(list(filter(lambda n: n != 0, map(lambda x: get_idx(x, word_to_idx), male_words))))
  female_idxs = np.array(list(filter(lambda n: n != 0, map(lambda x: get_idx(x, word_to_idx), female_words))))
  print(f'Number of male words: {male_idxs.shape[0]}, number of female words: {female_idxs.shape[0]}')
  return male_idxs, female_idxs

def partitioned_generate_cooccurrence_matrix(corpus, window_size, word_to_idx, partition_size):
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
            get_idx(word, word_to_idx),
            get_idx(context_word, word_to_idx),
          ] += 1 / abs(i - j)

  # row, col, value, gender
  co_matrix = np.zeros((len(co_occurrences_counter), 34))
  for n, ((id_word, id_context), value) in enumerate(co_occurrences_counter.items()):
    gender = get_gender(id_word, id_context)
    co_matrix[n] = np.array([id_word, id_context, value, gender])
  
  return co_matrix

def generate_cooccurrence_matrix(corpus, window_size, num_partitions):
  word_to_idx, idx_to_word = get_vocab_data(corpus)
  partition_size = len(corpus)//num_partitions

  print(f'Length of the corpus: {len(corpus)}, number of partitions selected: {num_partitions}')
  print(f'Partitions\' size (words): {partition_size}')

  co_occurrence_matrix = partitioned_generate_cooccurrence_matrix(corpus, window_size, word_to_idx, partition_size)
  
  # Save to file
  path = os.path.join(PATH, "cooccurrence_matrix_copy.pkl")
  with open(path, "wb+") as f:
    pickle.dump(co_occurrence_matrix, f)

  return co_occurrence_matrix

if __name__ == "__main__":
    corpus = load_text8()
    word_to_idx, idx_to_word = get_vocab_data(corpus)
    male_words, female_words = get_en_gender_sets()
    male_idxs, female_idxs = compute_gender_indexes(male_words, female_words, word_to_idx)
    print(male_idxs.shape, female_idxs.shape)