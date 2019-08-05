
class Node:
    def __init__(self, entry, all_nodes):
        self.base_path = entry['base_path']
        self.content_id = entry['content_id']
        self.title = entry['title']
        if 'parent_content_id' in entry:
            self.parent = all_nodes[entry['parent_content_id']]
            self.parent.children.append(self)
        else:
            self.parent = None
        self.children = []
    def embeddings_file_name(self):
        return re.sub(r'\W+', '', self.title) + ".pkl"

def get_embedded_sentences_for_taxon(content, taxon):
    # Calculating these is _slow_ so we store them on the hard drive
    # Open question as to whether we re-calculate these as stuff gets re-tagged
    # to a branch. I _suspect_ we shouldn't as we could end up subtly changing
    # the aboutness of a taxon, especially if there isn't much content there to begin
    # with, leading to poor overall outcome
    print("starting embedded sentences for taxon: " + taxon.title)
    filepath = labelled_file_path = os.path.join(ENCODINGSDATADIR, taxon.embeddings_file_name())
    if os.path.exists(filepath):
        print("encodings exist on hard drive")
        with open(filepath, 'rb') as fp:
            return pickle.load(fp)
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    content_ids_for_taxon = list(content_taxon_mapping[content_taxon_mapping['taxon_id'] == taxon.content_id]['content_id'])
    combined_text_for_content_in_taxon = list(content[content['content_id'].isin(content_ids_for_taxon)]['combined_text'])
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embedded_text_for_content_in_taxon = session.run(embed(combined_text_for_content_in_taxon))
    with open(filepath, 'wb') as fp:
        pickle.dump(embedded_text_for_content_in_taxon, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print("Embedded sentences for taxon and saved to file: " + taxon.title)
    return embedded_text_for_content_in_taxon

def get_score_for_item(content, embedded_sentences_for_taxon):
    content_generator = pairwise_distances_chunked(
        X=content,
        Y=embedded_sentences_for_taxon,
        working_memory=0,
        metric='cosine',
        n_jobs=-1)
    cosine_scores = list(enumerate(content_generator))[0][1][0]
    return cosine_scores.mean()

def get_content_to_retag(content, taxon):
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    content_ids_for_taxon = list(content_taxon_mapping[content_taxon_mapping['taxon_id'] == taxon.content_id]['content_id'])
    content_in_taxon = content[content['content_id'].isin(content_ids_for_taxon)]
    embedded_sentences_for_taxon = get_embedded_sentences_for_taxon(content, taxon)
    base_paths_for_content_to_retag = []
    for index, row in content_in_taxon.iterrows():
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            embedded_content = session.run(embed([row['combined_text']]))
            score = get_score_for_item(embedded_content, embedded_sentences_for_taxon)
            if score >= 0.65:
                base_paths_for_content_to_retag.append(row['base_path'])


import gzip
import ijson
import os
import pandas as pd
from sklearn.metrics import pairwise_distances_chunked
import re
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import operator
from sklearn.feature_extraction.text import TfidfVectorizer

DATADIR = os.getenv("DATADIR")
ENCODINGSDATADIR = DATADIR + "/encodings"
if DATADIR is None:
    print("You must set a DATADIR environment variable, see the readme in alphagov/govuk-taxonomy-supervised-learning repo for more details")
    sys.exit()

nodes = {}
taxons_path = os.path.join(DATADIR, 'taxons.json.gz')
with gzip.open(taxons_path, mode='rt') as input_file:
    taxons = ijson.items(input_file, prefix='item')
    for taxon in taxons:
        node = Node(taxon, nodes)
        nodes[node.content_id] = node


# Load in data
labelled_file_path = os.path.join(DATADIR, 'labelled.csv.gz')
labelled = pd.read_csv(labelled_file_path, compression='gzip', low_memory=False)

clean_content_path = os.path.join(DATADIR, 'clean_content.csv')
content = pd.read_csv(clean_content_path, low_memory=False)

# Load Universal Sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
embed = hub.Module(module_url)


# Lets just say that we have the relevant information
# Current taxon is the taxon we're trying to improve, here it's manually set to business-tax
current_taxon_id = "28262ae3-599c-4259-ae30-3c83a5ec02a1"
current_taxon = nodes[current_taxon_id]
# We're going to want to be able to populate this list automagically from the current taxon, shouldn't be too tricky
list_of_content_to_retag = ['/hmrc-internal-manuals/vat-womens-sanitary-products', '/guidance/changes-to-chief-commodity-codes-tariff-stop-press-notice-1','/guidance/rates-and-allowances-for-air-passenger-duty-historic-rates','/guidance/air-passenger-duty-and-connected-flights','/guidance/rates-and-allowances-for-air-passenger-duty','/guidance/poultry-from-iceland-tariff-quota-notice-73','/government/publications/iso-country-codes','/government/news/government-to-waive-vat-on-military-wives-charity-single','/guidance/laser-skin-treatment-and-hair-removal-tariff-notice-3','/government/collections/gwe-rwydo-a-sgamiau','/government/publications/notice-373-importing-visual-and-auditory-materials-free-of-duty-and-vat']
# Tried to calculate content to retag but it was SUPER SLOW so will come back to it later
# content_to_retag = get_content_to_retag(content, current_taxon)

current_scores = {}
best_fits = {}
suggested_taxons = {}
for content_to_retag_base_path in list_of_content_to_retag:
    print("attempting_to_retag:" + content_to_retag_base_path)
    combined_text = content[content['base_path'] == content_to_retag_base_path].iloc[0,:]['combined_text']
    embedded_content = None
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        embedded_content = session.run(embed([combined_text]))
    embedded_sentences_for_taxon = get_embedded_sentences_for_taxon(content, current_taxon)
    # Get the score of the content item against the taxon it's currently in
    current_score = get_score_for_item(embedded_content, embedded_sentences_for_taxon)
    current_scores[content_to_retag_base_path] = current_score
    # Get the score of the content item against all child items
    cosine_scores = {}
    for child_taxon in current_taxon.children:
        embedded_sentences_for_taxon = get_embedded_sentences_for_taxon(content, child_taxon)
        cosine_scores[child_taxon.title] = get_score_for_item(embedded_content, embedded_sentences_for_taxon)
    # Sort the scores
    cosine_scores = sorted(cosine_scores.items(), key=operator.itemgetter(1))
    best_fit = cosine_scores[0]
    best_fit_name = best_fit[0]
    best_fit_cosine_score = best_fit[1]
    # Calculate tf-idf score between content item title and taxon name
    vectorizer = TfidfVectorizer()
    vectorizer.fit([best_fit_name])
    content_item_title = content[content['base_path'] == content_to_retag_base_path].iloc[0,:]['title']
    tf_idf_score = vectorizer.transform([content_item_title]).mean()
    # Add to best fits
    best_fits[content_to_retag_base_path] = {'name': best_fit_name, 'cosine': best_fit_cosine_score, 'tf_idf': tf_idf_score}
    # See if it's suggested
    if best_fit_cosine_score < 0.5:
        suggested_taxons[content_to_retag_base_path] = {'name': best_fit_name, 'cosine': best_fit_cosine_score, 'tf_idf': tf_idf_score}
    if best_fit_cosine_score >= 0.5 and best_fit_cosine_score <= 0.6 and tf_idf_score > 0:
        suggested_taxons[content_to_retag_base_path] = suggested_taxons[content_to_retag_base_path] = {'name': best_fit_name, 'cosine': best_fit_cosine_score, 'tf_idf': tf_idf_score}

print("Current scores")
print(current_scores)

print("Best fit taxons")
print(best_fits)

print("Taxons we think we can retag to")
print(suggested_taxons)