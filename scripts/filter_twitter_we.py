import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from debiaswe.we import WordEmbedding
import tqdm

E = WordEmbedding('./embeddings/twitter128.tsv')

E.filter_words(lambda w: w.isalpha() and w.islower())

with open('./embeddings/filtered_twitter128.tsv', "w", encoding="utf-8") as file:
    for word in tqdm.tqdm(E.words, desc="Writing filtered embeddings: "):
        string_to_write = "\t".join([word] + list(map(str, list(E.v(word)))))+"\n"
        file.write(string_to_write, )
print("Done! Check the file ./embeddings/filtered_twitter128.tsv")