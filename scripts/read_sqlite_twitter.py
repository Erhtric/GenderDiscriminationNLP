"""
Read the twitter embeddings downloaded from http://www.italianlp.it/resources/italian-word-embeddings/
"""

import sqlite3
import sys
import codecs
import os
from tqdm import tqdm 

embedding_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embeddings")

outfile = codecs.open(os.path.join(embedding_path, "twitter128.tsv"), 'w', 'utf-8')
conn = sqlite3.connect(os.path.join(embedding_path, "twitter128.sqlite"))
c = conn.cursor()

vocab_size = c.execute("SELECT COUNT(*) from store").fetchone()[0]

for row in tqdm(c.execute("SELECT * from store"), "Extracting itWac embeddings: ", total=vocab_size):
  if row[0] == "\t":
    continue
  outfile.write("\t".join(str(x) for x in row[0:-1]) + "\n")