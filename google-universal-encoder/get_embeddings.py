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

if not os.path.exists('../combined_text_full'):
    os.makedirs('../combined_text_full')

LOG_DIR = "../combined_text_full"
path_for_metadata = os.path.join(LOG_DIR,'metadata.tsv')
DATADIR = os.getenv("DATADIR")

print("read in labelled")
labelled = pd.read_csv(os.path.join(DATADIR, 'labelled.csv.gz'), compression='gzip', low_memory=False)
taxon_id_to_base_path = dict(zip(labelled['taxon_id'], labelled['taxon_base_path']))
labelled['brexit'] = np.where(labelled['level2taxon']=='Brexit', 1, 0)



# ### Prepare data
corpus_sample = labelled.sample(n=2000, random_state=1234)
print("get short string data")
corpus = labelled['combined_text'].tolist()

TEXT_LENGTH = 300
short_corpus=[]
for text in corpus:
    words = text.split()
    truncated = " ".join(words[0:TEXT_LENGTH])
    short_corpus.append(truncated)


# ## Save out data for tensorboard embeddings projector visualisation
# These are things that you can label or color the points (content items) by
print("save out metadata for tensorboard projector")
with open(path_for_metadata,'w') as f:
    f.write("Index\tTitle\tTaxon1\tTaxon2\tbrexit\n")
    for index, row in corpus_sample.iterrows():
        f.write("{}\t{}\t{}\t{}\t{}\n".format(index,row['title'], row['level1taxon'],row['level2taxon'], row['brexit']))


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

with tf.Session() as sess:
    # for tensorboard
    emb = tf.Variable(embedded_sentences, name='embedded_sentences')
    sess.run(emb.initializer)
    config = projector.ProjectorConfig()
    summary_writer = tf.summary.FileWriter(LOG_DIR)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = emb.name

    # Comment out if you don't have metadata
    embedding.metadata_path = path_for_metadata

    projector.visualize_embeddings(summary_writer, config)
    saver = tf.train.Saver([emb])
    saver.save(sess, os.path.join(LOG_DIR, 'combined_text_sample_full.ckpt'), 1)
    print("Model saved in path: %s" % os.path.join(LOG_DIR, 'combined_text_sample_full.ckpt'))


# ### Save out embedding vectors
print("save out embeddings")
np.save('embedded_sentences'+os.path.basename(DATADIR)+'.npy', embedded_sentences)
