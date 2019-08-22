#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow_hub as hub
import os
import pandas as pd

DATADIR = os.getenv("DATADIR")

print("read in clean_content")
clean_content = pd.read_csv(os.path.join(DATADIR, 'clean_content.csv'), low_memory=False)

# Prepare data
corpus = clean_content['combined_text'].tolist()

TEXT_LENGTH = 512
short_corpus=[]
for text in corpus:
    words = text.split()
    truncated = " ".join(words[0:TEXT_LENGTH])
    short_corpus.append(truncated)

# Generate embeddings
# Import the Universal Sentence Encoder's TF Hub module
print("download the hub module")
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(module_url)

print("run the model")
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    embedded_sentences = session.run(embed(short_corpus))

# Combine embedded sentences with clean_content
clean_content['embedded_sentences'] = embedded_sentences.tolist()

# Save out content with embeddings
print("save out embeddings")
clean_content.to_pickle(DATADIR + '/embedded_clean_content.pkl')
