
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
    def recursive_children(self):
        if self.children:
            results = []
            for child in self.children:
                results.append(child.recursive_children())
            return [item for sublist in results for item in sublist]
        else:
            return [self]
    def all_siblings_and_children(self):
        results = []
        for node in self.parent.children:
            results.append(node.recursive_children())
        flattened_results = [item for sublist in results for item in sublist]
        # Remove self from results
        return [result for result in flattened_results if result.content_id != self.content_id]


class Tree:
    def __init__(self, datadir):
        self.nodes = {}
        taxons_path = os.path.join(datadir, 'taxons.json.gz')
        with gzip.open(taxons_path, mode='rt') as input_file:
            taxons = ijson.items(input_file, prefix='item')
            for taxon in taxons:
                node = Node(taxon, self.nodes)
                self.nodes[node.content_id] = node
    def find(self, content_id):
        return self.nodes[content_id]

def get_embedded_sentences_for_taxon(content, taxon):
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    content_ids_for_taxon = list(content_taxon_mapping[content_taxon_mapping['taxon_id'] == taxon.content_id]['content_id'])
    return content[content['content_id'].isin(content_ids_for_taxon)]['embedded_sentences']

def get_score_for_item(content, embedded_sentences_for_taxon):
    embedded_sentences_for_taxon_list = embedded_sentences_for_taxon.to_list()
    if not embedded_sentences_for_taxon_list:
        return float("inf")
    content_generator = pairwise_distances_chunked(
        X=[content],
        Y=embedded_sentences_for_taxon_list,
        working_memory=0,
        metric='cosine',
        n_jobs=-1)
    cosine_scores = list(enumerate(content_generator))[0][1][0]
    return cosine_scores.mean()

import gzip
import ijson
import os
import pandas as pd
from sklearn.metrics import pairwise_distances_chunked
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

DATADIR = os.getenv("DATADIR")
ENCODINGSDATADIR = DATADIR + "/encodings"
if DATADIR is None:
    print("You must set a DATADIR environment variable, see the readme in alphagov/govuk-taxonomy-supervised-learning repo for more details")
    sys.exit()

tree = Tree(DATADIR)

# Load in data
labelled_file_path = os.path.join(DATADIR, 'labelled.csv.gz')
labelled = pd.read_csv(labelled_file_path, compression='gzip', low_memory=False)

clean_content_path = os.path.join(DATADIR, 'embedded_clean_content.pkl')
content = pd.read_pickle(clean_content_path)

problem_content_path = os.path.join(DATADIR, 'problem_content.csv')
problem_content = pd.read_csv(problem_content_path)

current_scores = {}
best_fits = {}
suggested_taxons = {}
for index, row in problem_content.iterrows():
    content_to_retag_base_path = row["base_path"]
    current_taxon = tree.find(row["taxon_id"])
    if not current_taxon.all_siblings_and_children():
        # No children or siblings in the same branch for current taxon, at moment just leave it there
        print("No siblings or children of current taxon, leaving in place: " + content_to_retag_base_path)
        suggested_taxons[content_to_retag_base_path] = { 'current_taxon': current_taxon.title, 'new_taxon': 'No children!', 'cosine': '?', 'tf_idf': '?' }
    else:
        print("attempting_to_retag: " + content_to_retag_base_path)
        embedded_content = content[content['base_path'] == content_to_retag_base_path].iloc[0,:]['embedded_sentences']
        embedded_sentences_for_taxon = get_embedded_sentences_for_taxon(content, current_taxon)
        # Get the score of the content item against the taxon it's currently in
        current_score = get_score_for_item(embedded_content, embedded_sentences_for_taxon)
        current_scores[content_to_retag_base_path] = current_score
        # Get the score of the content item against all child items
        cosine_scores = {}
        for taxon in current_taxon.all_siblings_and_children():
            embedded_sentences_for_taxon = get_embedded_sentences_for_taxon(content, taxon)
            cosine_scores[taxon.title] = get_score_for_item(embedded_content, embedded_sentences_for_taxon)
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
        best_fits[content_to_retag_base_path] = {'current_taxon': current_taxon.title, 'new_taxon': best_fit_name, 'cosine': best_fit_cosine_score, 'tf_idf': tf_idf_score}
        # See if it's suggested
        if best_fit_cosine_score < 0.5:
            suggested_taxons[content_to_retag_base_path] = {'current_taxon': current_taxon.title, 'new_taxon': best_fit_name, 'cosine': best_fit_cosine_score, 'tf_idf': tf_idf_score}
        if best_fit_cosine_score >= 0.5 and best_fit_cosine_score <= 0.6 and tf_idf_score > 0:
            # For the ~27 items suggested by this process, almost all have been false positives so we'll probably want to remove it
            suggested_taxons[content_to_retag_base_path] = suggested_taxons[content_to_retag_base_path] = {'current_taxon': current_taxon.title, 'new_taxon': best_fit_name, 'cosine': best_fit_cosine_score, 'tf_idf': tf_idf_score}

with open("best_fits.csv", 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(['base_path', 'current_taxon', 'new_taxon', 'cosine', 'tf_idf'])
    for content_base_path, results in best_fits.items():
        filewriter.writerow([content_base_path, results['current_taxon'], results['new_taxon'], results['cosine'], results['tf_idf']])

with open("suggested_retaggings.csv", 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(['base_path', 'current_taxon', 'new_taxon', 'cosine', 'tf_idf'])
    for content_base_path, results in suggested_taxons.items():
        filewriter.writerow([content_base_path, results['current_taxon'], results['new_taxon'], results['cosine'], results['tf_idf']])