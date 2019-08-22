
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
        self.all_sibs_and_children = None
    def unique_title(self):
        # Some taxa have identical names so using the title as the
        # key of a dictionary will cause problems
        return self.content_id[:3] + " " + self.title
    def title_and_parent_title(self):
        if self.parent:
            return self.parent.title + "/" + self.title
        else:
            return self.title
    def recursive_children(self):
        if self.children:
            results = []
            for child in self.children:
                results.append(child.recursive_children())
            return [item for sublist in results for item in sublist]
        else:
            return [self]
    def all_siblings_and_children(self):
        if self.all_sibs_and_children is None:
            results = []
            # This is a slightly hacky way of not returning the entire tree if the node
            # is a level 1 taxon. E.g. if the taxon has no parent it's level 1 so only return it's children
            # rather than all it's siblings and their children (which would be the entire tree)
            if not self.parent:
                results.append(self.recursive_children())
            else:
                for node in self.parent.children:
                    results.append(node.recursive_children())
            flattened_results = [item for sublist in results for item in sublist]
            # Remove self from results
            self.all_sibs_and_children = [result for result in flattened_results if result.content_id != self.content_id]
            return self.all_sibs_and_children
        else:
            return self.all_sibs_and_children


class Tree:
    def __init__(self, datadir):
        self.nodes = {}
        taxons_path = os.path.join(datadir, 'taxons.json.gz')
        with gzip.open(taxons_path, mode='rt') as input_file:
            taxons = ijson.items(input_file, prefix='item')
            for taxon in taxons:
                node = Node(taxon, self.nodes)
                self.nodes[node.content_id] = node
    def find(self, taxon_content_id):
        return self.nodes[taxon_content_id]

# This identifies content items in a taxon that might be out of place
# Content items are flagged if their mean score is above a particular threshold (default 0.65).
def get_misplaced_content_in_taxon(content, taxon, similarity_threshold = 0.65):
    print('Looking for misplaced content in taxon: ', taxon.title)
    taxon_embeddings = get_embedded_sentences_for_taxon(content, taxon)
    distances_between_all_content_item_pairs = pairwise_distances(
        taxon_embeddings,
        metric = 'cosine',
        n_jobs = -1
    )
    content_for_taxon = get_content_for_taxon(content, taxon).copy()
    content_for_taxon['mean_cosine_score'] = distances_between_all_content_item_pairs.mean(axis=1)
    misplaced_items = content_for_taxon.loc[content_for_taxon['mean_cosine_score'] > similarity_threshold].copy()
    misplaced_items["taxon_id"] = taxon.content_id
    misplaced_items["taxon_title"] = taxon.unique_title()
    return misplaced_items;

# Finds all content that might be incorrectly tagged
# Currently hard coded to look in money branch but could look anywhere
def find_misplaced_items(content, tree):
    # Content id for money node
    money_branch_apex_node = tree.find("6acc9db4-780e-4a46-92b4-1812e3c2c48a")
    taxons_to_search = [money_branch_apex_node] + money_branch_apex_node.all_siblings_and_children()
    misplaced_items = pd.DataFrame()
    for taxon in taxons_to_search:
        misplaced_items_for_taxon = get_misplaced_content_in_taxon(content, tree.find(taxon.content_id))
        misplaced_items = misplaced_items.append(misplaced_items_for_taxon)
    problem_content_path = os.path.join(DATADIR, 'problem_content.csv')
    print("Found " + str(len(misplaced_items)) + " misplaced items. Saving csv to " + problem_content_path)
    misplaced_items.to_csv(problem_content_path)
    return misplaced_items;

def get_embedded_sentences_for_taxon(content, taxon):
    return get_content_for_taxon(content, taxon)['combined_text_embedding'].to_list()

def get_content_for_taxon(content, taxon):
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    content_ids_for_taxon = list(content_taxon_mapping[content_taxon_mapping['taxon_id'] == taxon.content_id]['content_id'])
    return content[content['content_id'].isin(content_ids_for_taxon)];

def get_score_for_item(content, all_content, taxon):
    embedded_sentences_for_taxon = get_embedded_sentences_for_taxon(all_content, taxon)
    if not embedded_sentences_for_taxon:
        return [], -1;
    content_generator = pairwise_distances_chunked(
        X=[content],
        Y=embedded_sentences_for_taxon,
        working_memory=0,
        metric='cosine',
        n_jobs=-1)
    cosine_scores = list(enumerate(content_generator))[0][1][0]
    cosine_scores.sort()
    return cosine_scores, cosine_scores.mean();

def get_cosine_scores_for_sibling_and_children_taxons(current_taxon, embedded_content, content):
    mean_cosine_scores_for_each_taxon = {}
    all_cosine_scores_for_each_taxon = {}
    for i, taxon in enumerate(current_taxon.all_siblings_and_children()):
        all_scores, mean = get_score_for_item(embedded_content, content, taxon)
        if mean > -1:
            all_cosine_scores_for_each_taxon[taxon.unique_title()] = all_scores
            mean_cosine_scores_for_each_taxon[taxon.unique_title()] = mean
    mean_cosine_score_for_each_taxon = sorted(mean_cosine_scores_for_each_taxon.items(), key=operator.itemgetter(1))
    return (mean_cosine_score_for_each_taxon, all_cosine_scores_for_each_taxon);

# This was an attempt at a better scoring system to get around the fact that mean isn't so great
def get_distance_cosine_scores(mean_cosine_score_for_each_taxon, all_cosine_scores_for_each_taxon):
    distance_cosine_score_for_each_taxon = {}
    for i, scores in enumerate(mean_cosine_score_for_each_taxon):
        unique_taxon_title = scores[0]
        all_content_scores = all_cosine_scores_for_each_taxon[unique_taxon_title]
        total_distance = 0
        for content_score in all_content_scores:
            total_distance += (1 + content_score) * content_score
        if total_distance > 0:
            total_distance /= len(all_content_scores)
        else:
            total_distance = float('inf')
        distance_cosine_score_for_each_taxon[unique_taxon_title] = total_distance
    distance_cosine_score_for_each_taxon = sorted(distance_cosine_score_for_each_taxon.items(), key=operator.itemgetter(1))
    if len(distance_cosine_score_for_each_taxon) >= 1:
        best_fit = distance_cosine_score_for_each_taxon[0]
        return (best_fit[0], best_fit[1], distance_cosine_score_for_each_taxon)
    else:
        return (None, None, distance_cosine_score_for_each_taxon);

def calculate_distance(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Decides whether the tagging to current_taxon can be removed
# Returns two booleans and debugging info
#  - Whether it can be retagged
#  - Whether it's decision requires human confirmation
#  - A string explaining why it doesn't need to be untagged, or if it does, a dictionary of all the cosine scores of all the taggings
def can_be_untagged(current_taxon, content, content_to_retag_base_path):
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    taxon_ids_for_content = list(content_taxon_mapping[content_taxon_mapping['content_id'] == "0fea02ed-c1c8-4502-a7a2-f0ebebe1ee1c"]['taxon_id'])
    if len(taxon_ids_for_content) <= 1:
        # No other taggings so we can't untag
        return (False, False, "Has no other taggings so we cannot untag");
    # See if there is a tagging below the current taxon
    for child in current_taxon.recursive_children():
        if child.content_id in taxon_ids_for_content:
            return (True, False, "Tagged to taxon below the current one so we can untag without human intervention");
    return (True, True, {})
    # The below was an attempt to see if the other taggings are better than the current one.
    # This may not be necessary as if this tag has been flagged as possibly incorrect then we
    # ought to be deleting it by default. This is commented out for interest and because it might
    # be useful
    # current_scores = {}
    # embedded_content = content[content['base_path'] == content_to_retag_base_path].iloc[0,:]['combined_text_embedding']
    # for taxon_id in taxon_ids_for_content:
    #     taxon = tree.find(taxon_id)
    #     # embedded_sentences_for_taxon = get_embedded_sentences_for_taxon(content, taxon)
    #     # Get the score of the content item against the taxon it's currently in
    #     scores, mean = get_score_for_item(embedded_content, content, taxon)
    #     current_scores[taxon.unique_title()] = mean
    # print(current_scores)
    # _all_scores_for_current_taxon, score_for_current_taxon = get_score_for_item(embedded_content, content, current_taxon)
    # scores_for_all_taxons = list(current_scores.values()).sort()
    # # Not strictly speaking a median but...
    # if scores_for_all_taxons is None:
    #     return (False, False, "No cosine similarity scores")
    # median = scores_for_all_taxons[len(scores_for_all_taxons) / 2]
    # if score_for_current_taxon >= median:
    #     return (True, True, current_scores)
    # else:
    #     return (False, True, current_scores)

def debugging_entry(base_path, current_taxon, debugging_info):
    return {
        'base_path': base_path,
        'current_taxon_title': current_taxon.title,
        'current_taxon_content_id': current_taxon.content_id,
        'debugging_info': debugging_info
    }

def untagging_entry(problem_content_row, current_taxon, more_info):
    return [problem_content_row["base_path"], problem_content_row["content_id"], current_taxon.title_and_parent_title(), current_taxon.content_id, more_info]

import gzip
import ijson
import os
import pandas as pd
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
import operator
import numpy as np
from sklearn.cluster import KMeans
import math
import csv

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

# Load misplaced items
problem_content = find_misplaced_items(content, tree)

content_to_retag = []
content_for_human_verification_to_untag = []
content_to_untag = []
debugging_info = []
for index, row in problem_content.iterrows():
    content_to_retag_base_path = row["base_path"]
    current_taxon = tree.find(row["taxon_id"])
    print(content_to_retag_base_path)
    if not current_taxon.all_siblings_and_children():
        # No children or siblings in the same branch for current taxon, at moment just leave it there
        debugging_info.append(debugging_entry(content_to_retag_base_path, current_taxon, "No siblings or children of current taxon"))
        next()
    should_be_untagged, requires_human_confirmation, more_info = can_be_untagged(current_taxon, content, content_to_retag_base_path)
    if should_be_untagged:
        if requires_human_confirmation:
            content_for_human_verification_to_untag.append(untagging_entry(row, current_taxon, more_info))
        else:
            content_to_untag.append(untagging_entry(row, current_taxon, more_info))
            next()
    print("Attempting_to_retag: " + content_to_retag_base_path)
    embedded_content = content[content['base_path'] == content_to_retag_base_path].iloc[0,:]['combined_text_embedding']
    # Get the score of the content item against all items
    mean_cosine_score_for_each_taxon, all_cosine_scores_for_each_taxon = get_cosine_scores_for_sibling_and_children_taxons(current_taxon, embedded_content, content)
    best_distance_suggestion, best_distance_cosine_score, distance_cosine_scores = get_distance_cosine_scores(mean_cosine_score_for_each_taxon, all_cosine_scores_for_each_taxon)
    if best_distance_cosine_score is not None:
        content_to_retag.append([content_to_retag_base_path, current_taxon.title_and_parent_title(), best_distance_suggestion, best_distance_cosine_score, distance_cosine_scores])

with open("content_to_retag.csv", 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(['content_to_retag_base_path', 'current_taxon', 'best_distance_suggestion', 'best_distance_cosine_score', 'distance_cosine_scores'])
    for row in content_to_retag:
        filewriter.writerow(row)

with open("content_for_human_verification_to_untag.csv", 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(["content_to_untag_base_path", "content_to_untag_content_id", "current_taxon_name", "current_taxon_content_id", "more_info"])
    for row in content_for_human_verification_to_untag:
        filewriter.writerow(row)

with open("content_to_untag.csv", 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(["content_to_untag_base_path", "content_to_untag_content_id", "current_taxon_name", "current_taxon_content_id", "more_info"])
    for row in content_to_untag:
        filewriter.writerow(row)