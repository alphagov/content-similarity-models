# This may or may not run without intervention but should get you 90% if the way there
# It's fairly slow but could be speeded up

import gzip
import ijson
import os
import pandas as pd
from sklearn.metrics import pairwise_distances_chunked
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import math
import csv
from sklearn.manifold import TSNE
import altair as alt
from scipy import interpolate

DATADIR = os.getenv("DATADIR")
if DATADIR is None:
    print("You must set a DATADIR environment variable, see the readme in alphagov/govuk-taxonomy-supervised-learning repo for more details")
    sys.exit()

tree = Tree(DATADIR)

# Load in data
labelled_file_path = os.path.join(DATADIR, 'labelled.csv.gz')
labelled = pd.read_csv(labelled_file_path, compression='gzip', low_memory=False)

clean_content_path = os.path.join(DATADIR, 'embedded_clean_content.pkl')
content = pd.read_pickle(clean_content_path)

def get_content_for_taxon(content, taxon):
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    content_ids_for_taxon = list(content_taxon_mapping[content_taxon_mapping['taxon_id'] == taxon.content_id]['content_id'])
    return content[content['content_id'].isin(content_ids_for_taxon)]

def get_scores_for_taxon(all_content, taxon):
    embedded_sentences_for_taxon = get_embedded_sentences_for_taxon(all_content, taxon)
    if not embedded_sentences_for_taxon:
        return [], -1;
    content_generator = pairwise_distances_chunked(
        X=embedded_sentences_for_taxon,
        Y=embedded_sentences_for_taxon,
        working_memory=0,
        metric='cosine',
        n_jobs=-1)
    cosine_scores = list(enumerate(content_generator))[0][1][0]
    cosine_scores.sort()
    return cosine_scores

# scores = get_scores_for_taxon(content, tree.find("d6c2de5d-ef90-45d1-82d4-5f2438369eea"))
content_for_taxon = get_content_for_taxon(content, tree.find("d6c2de5d-ef90-45d1-82d4-5f2438369eea"))
tsne = TSNE(n_components=2, verbose=1, perplexity=20, n_iter=7000)
tsne_results = tsne.fit_transform(content_for_taxon['embedded_sentences'].to_list())
new_tsne_results = tsne_results.copy()

def calculate_distance(x1,y1,x2,y2):
    if x1 == x2 and y1 == y2:
        return 0
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

for _ in range (10):
    for (index, coord) in enumerate(tsne_results):
        x = coord[0]
        y = coord[1]
        distances = []
        for other_coord in tsne_results:
            distance = calculate_distance(x, y, other_coord[0], other_coord[1])
            if distance > 0:
                # Strength is inverse square of distance
                new_distance = 1 * (1 / distance)
                result = {
                    'distance': distance,
                    'new_distance': new_distance,
                    'coords': other_coord
                }
                distances.append(result)
        distances = sorted(distances, key = lambda i: i['distance'])
        for other_coord in distances:
            other_x = other_coord['coords'][0]
            other_y = other_coord['coords'][1]
            x += (other_coord['new_distance'] / other_coord['distance']) * (other_x - x)
            y += (other_coord['new_distance'] / other_coord['distance']) * (other_y - y)
        new_tsne_results[index] = [x, y]

df = labelled[labelled['taxon_id']=='d6c2de5d-ef90-45d1-82d4-5f2438369eea'].copy()
df['x-tsne'] = tsne_results[:,0]
df['y-tsne'] = tsne_results[:,1]
df['new-x-tsne'] = new_tsne_results[:,0]
df['new-y-tsne'] = new_tsne_results[:,1]

brexit_tsne = alt.Chart(df[['x-tsne', 'y-tsne','title']]).mark_circle(size=50).encode(
    alt.X('x-tsne:Q',
          axis=alt.Axis(grid=False)),
    alt.Y('y-tsne:Q',
          axis=alt.Axis(grid=False)),
    tooltip=['title:N']
).configure_axis(
    titleFontSize=15).interactive()

brexit_tsne_new = alt.Chart(df[['new-x-tsne', 'new-y-tsne','title']]).mark_circle(size=10).encode(
    alt.X('new-x-tsne:Q',
          axis=alt.Axis(grid=False)),
    alt.Y('new-y-tsne:Q',
          axis=alt.Axis(grid=False)),
    tooltip=['title:N']
).configure_axis(
    titleFontSize=15).interactive()

brexit_tsne.serve()
brexit_tsne_new.serve()