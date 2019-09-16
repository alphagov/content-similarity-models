import gzip
import ijson
import os
import pandas as pd
from sklearn.metrics import pairwise_distances
import csv
from nltk.stem.porter import PorterStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

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
    def recursive_parents(self):
        results = []
        if self.parent:
            results.append(self.parent.recursive_parents())
        else:
            results.append([self])
        # Set to make them unique
        flattened = flatten(results)
        unique = list(set(flattened))
        return unique;
    def title_and_parent_title(self):
        if self.parent is not None:
            return self.parent.title + " ... > ... " + self.title
        else:
            return self.title;
    def recursive_children(self):
        results = []
        results.append([self])
        for child in self.children:
            results.append(child.recursive_children())
        # Set to make them unique
        flattened = flatten(results)
        unique = list(set(flattened))
        return unique;
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
            flattened_results = flatten(results)
            # Remove self from results
            self.all_sibs_and_children = [result for result in flattened_results if result.content_id != self.content_id]
            return self.all_sibs_and_children
        else:
            return self.all_sibs_and_children
    def is_apex(self):
        return self.parent is None


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
    def apex_nodes(self):
        apex_nodes = []
        for node in self.nodes.values():
            if node.is_apex():
                apex_nodes.append(node)
        return apex_nodes

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

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
def find_misplaced_items(apex_node, content, tree):
    taxons_to_search = [apex_node] + apex_node.recursive_children()
    misplaced_items = pd.DataFrame()
    for taxon in taxons_to_search:
        misplaced_items_for_taxon = get_misplaced_content_in_taxon(content, tree.find(taxon.content_id), 0.6)
        misplaced_items = misplaced_items.append(misplaced_items_for_taxon)
    unique_misplaced_items = misplaced_items.drop_duplicates(subset=['content_id','taxon_id'])
    problem_content_path = os.path.join(DATADIR, f"problem_content_#{apex_node.title}.csv")
    print("Found " + str(len(unique_misplaced_items)) + " misplaced items. Saving csv to " + problem_content_path)
    unique_misplaced_items.to_csv(problem_content_path)
    return unique_misplaced_items;

def get_embedded_sentences_for_taxon(content, taxon):
    return get_content_for_taxon(content, taxon)['combined_text_embedding'].to_list()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def get_embedded_titles_for_taxon(content, taxon):
    return get_content_for_taxon(content, taxon)['title_embedding'].to_list()

def get_content_for_taxon(content, taxon):
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    content_ids_for_taxon = list(content_taxon_mapping[content_taxon_mapping['taxon_id'] == taxon.content_id]['content_id'])
    return content[content['content_id'].isin(content_ids_for_taxon)];

models = {} # global variable
vectorizers = {}

def get_score_for(current_node, child_taxons, content_item_content, empty_arry, content):
    global models
    global vectorizers
    content_ids = [taxon.content_id for taxon in child_taxons]
    content_ids.sort()
    key = ",".join(content_ids)
    if key not in models:
        texts = []
        y = []
        for child_taxon in child_taxons:
            if child_taxon.content_id == current_node.content_id:
                # Ignore taxon if its the same as the current one
                print("Ignoring " + child_taxon.title + " as its the same as " + current_node.title)
                continue
            length_of_content = []
            print("looking at all children of " + child_taxon.title)
            for taxon in child_taxon.recursive_children():
                print("Those children include: " + taxon.title)
                for i, content_item in get_content_for_taxon(content, taxon).iterrows():
                    texts.append(content_item['combined_text'])
                    y.append(child_taxon.content_id)
                    length_of_content.append(len(tokenize(content_item['combined_text'])))
        length_of_content.sort()
        if len(list(set(y))) <= 1 or len(length_of_content) == 0:
            print("One or fewer classes, returning early")
            return (False, False);
        print("Average length of page for taxon: " + str(sum(length_of_content) / len(length_of_content) ))
        median_length_of_page = length_of_content[int(len(length_of_content) / 2)]
        print("Median length of page for taxon: " + str(median_length_of_page))
        max_features = median_length_of_page
        vectorizer = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words='english', max_features=max_features )
        X = vectorizer.fit_transform(texts).toarray()
        vectorizers[key] = vectorizer
        model = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=200).fit(X, y)
        models[key] = model
    else:
        model = models[key]
        vectorizer = vectorizers[key]
    prediction = model.predict(vectorizer.transform([content_item_content]))[0]
    probability = model.predict_proba(vectorizer.transform([content_item_content]))[0][0]
    return (prediction, probability);

# Decides whether the tagging to current_taxon can be removed
# Returns two booleans and debugging info
#  - Whether it can be retagged
#  - Whether it's decision requires human confirmation
#  - A string explaining why it doesn't need to be untagged, or if it does, a dictionary of all the cosine scores of all the taggings
def can_be_untagged(current_taxon, content, content_to_retag_content_id):
    content_taxon_mapping_path = os.path.join(DATADIR, 'content_to_taxon_map.csv')
    content_taxon_mapping = pd.read_csv(content_taxon_mapping_path, low_memory=False)
    taxon_ids_for_content = list(content_taxon_mapping[content_taxon_mapping['content_id'] == content_to_retag_content_id]['taxon_id'])
    if len(taxon_ids_for_content) <= 1:
        # No other taggings so we can't untag
        return (False, False, "Has no other taggings so we cannot untag");
    # See if there is a tagging below the current taxon
    for child in current_taxon.recursive_children():
        if child.content_id in taxon_ids_for_content:
            return (True, False, "Tagged to taxon below the current one so we can untag without human intervention");
    if len(taxon_ids_for_content) > 1:
        return (True, True, {})

def debugging_entry(base_path, current_taxon, debugging_info):
    return {
        'base_path': base_path,
        'current_taxon_title': current_taxon.title,
        'current_taxon_content_id': current_taxon.content_id,
        'debugging_info': debugging_info
    }


DATADIR = os.getenv("DATADIR")
if DATADIR is None:
    print("You must set a DATADIR environment variable, see the readme in alphagov/govuk-taxonomy-supervised-learning repo for more details")
    sys.exit()

tree = Tree(DATADIR)

# Load in data
clean_content_path = os.path.join(DATADIR, 'embedded_clean_content.pkl')
content = pd.read_pickle(clean_content_path)
apex_node = tree.find("c58fdadd-7743-46d6-9629-90bb3ccc4ef0")

# Load misplaced items
problem_content = find_misplaced_items(apex_node, content, tree)

# global models
# global vectorizers
models = {}
vectorizers = {}
content_to_retag = []
content_for_human_verification_to_untag = []
content_to_untag = []
debugging_info = []
for index, row in problem_content.iterrows():
    content_to_retag_base_path = row["base_path"]
    current_taxon = tree.find(row["taxon_id"])
    if not current_taxon.all_siblings_and_children():
        # No children or siblings in the same branch for current taxon, at moment just leave it there
        debugging_info.append(debugging_entry(content_to_retag_base_path, current_taxon, "No siblings or children of current taxon"))
        continue
    should_be_untagged, requires_human_confirmation, more_info = can_be_untagged(current_taxon, content, row["content_id"])
    if should_be_untagged:
        if requires_human_confirmation:
            content_for_human_verification_to_untag.append([content_to_retag_base_path, current_taxon.title, current_taxon.base_path, more_info])
        else:
            content_to_untag.append([content_to_retag_base_path, current_taxon.title, current_taxon.base_path, more_info])
            continue
    print("Attempting_to_retag: " + content_to_retag_base_path)
    content_item_content = content[content['base_path'] == content_to_retag_base_path].iloc[0,:]['combined_text']
    # Get the score of the current taxon so we can see if it's children do any better
    scores_for_current_taxon = -1
    if scores_for_current_taxon:
        best_score = scores_for_current_taxon
        node = apex_node
        while any(node.children):
            print("Looking at children of " + node.title)
            taxon_content_id, probability = get_score_for(node, node.children, content_item_content, [], content)
            if not taxon_content_id:
                # There are no scores, so none were relevant, we can break
                break
            else:
                best_taxon = tree.find(taxon_content_id)
                print("Best taxon is: " + best_taxon.title)
                print("Which has probability score of: " + str(probability) + "%")
                best_score = probability
                node = best_taxon
        if node is not apex_node and node is not current_taxon:
            content_to_retag.append([content_to_retag_base_path, current_taxon.title_and_parent_title(), current_taxon.base_path, node.content_id, node.title, node.title_and_parent_title(), node.base_path, best_score, []])

with open("content_to_retag_" + apex_node.title + ".csv", 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(['content_to_retag_base_path', 'current_taxon_title', 'current_taxon_base_path', 'suggestion_content_id', 'suggestion_title', 'suggestion_title_and_level_1', 'suggestion_base_path', 'suggestion_cosine_score', 'other_suggestions'])
    for row in content_to_retag:
        filewriter.writerow(row)

with open("depth_first_content_for_human_verification_to_untag_" + apex_node.title + ".csv", 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(['content_to_retag_base_path', "current_taxon", "current_taxon_base_path", "more_info"])
    for row in content_for_human_verification_to_untag:
        filewriter.writerow(row)

with open("depth_first_content_to_untag_" + apex_node.title + ".csv", 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(['content_to_retag_base_path', "current_taxon", "current_taxon_base_path", "more_info"])
    for row in content_to_untag:
        filewriter.writerow(row)