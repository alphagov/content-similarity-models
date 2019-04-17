#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.tensorboard.plugins import projector

import numpy as np
import os
import pandas as pd

# ### Set-up and get data


module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" 

DATADIR = os.getenv("DATADIR")

print("read in clean_content")
clean_content = pd.read_csv(os.path.join(DATADIR, 'clean_content.csv'), low_memory=False)

# ### Prepare data
corpus = clean_content['combined_text'].tolist()

TEXT_LENGTH = 512
short_corpus=[]
for text in corpus:
    words = text.split()
    truncated = " ".join(words[0:TEXT_LENGTH])
    short_corpus.append(truncated)


# ## Generate embeddings
# Import the Universal Sentence Encoder's TF Hub module
print("download the hub module")
embed = hub.Module(module_url)

# Reduce logging output.
# tf.logging.set_verbosity(tf.logging.ERROR)
print("run the model")
with tf.Session() as session:
    
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    embedded_sentences = session.run(embed(short_corpus))
#     session.run(embed(corpus))


# ### Save out embedding vectors
print("save out embeddings")
np.save('embedded_clean_content'+os.path.basename(DATADIR)+'.npy', embedded_sentences)
