# Using the google universal sentence encoder on GOV.UK

The code in this directory uses
 [Google's universal sentence encoder](https://tfhub.dev/google/universal-sentence-encoder/2) to generate 512D vectors 
 for each content item. 
 
 These vectors have been used both to provide related links for AB tests on GOV.UK and to derive a content diversity
 score to help understand the health of the taxonomy and how to fix it.
 
 ## Embedding GOV.UK content items
 
 At the moment the script to do this has the labelled.csv.gz input  and sentence length of 300 hard-coded into it. 
 Future work would be to enable giving the data source and text length as an arg to the get_embeddings.py script.
 
 To run the script as it is now, ensure environment variables LOG_DIR and DATADIR are set and that labelled.csv.gz 
 is in DATADIR.
 
 It will output embedded content items in DATADIR, named embedded_sentences + date + npy. These are then read into the 
 notebooks.
 
 In addition, it will create metadata (including the taxons for each content item) and 
 tensorboard checkpoints in the LOG_DIR. These can be used with the tensorboard embedding projector to visualise
 individual content items in 3D using PCA or tSNE. 
 
 To see these, in your console, with activated environment:
 `tensorboard --logdir=universal_embeddings`
 
 then in your browser go to:
 `localhost:6006`
 
 ## Getting links for GOV.UK AB test
 
 The notebook get_links.ipynb contains the code to get links for the experiment as well as how to sample links based 
 on the publisher publishing_app (proxy for mainstream!) or using an input list of urls for sensitive content. These were
 used by content people to check the rationality of links and assess the risk of using these links on the site.
 
 There are various ways that the links can be selected e.g., top n links or beneath a threshold for cosine distance.
 
 To check in future, why the threshold generator is repeating content_ids in the suggested links. This was discovered once
 the experiment was live
 
 If using data other than labelled, some work will be need to be done to make this notebook adapt to different data.
 
 ## Getting diversity scores
 
 In the notebook get_homogeneity_scores_taxon.ipynb there is the code to calculate the mean distance between all pairs
 in a group of content, specified by taxon.
 
 The notebook also contains a set of charts to understand this diversity score in relation to taxon size and depth in the
 taxonomy tree (level 1 = branch). 
 
 These results have been narrated in a 
 [some google slides](https://docs.google.com/presentation/d/1dV2WK2wf4W7ElKRnYHDr57yo_AIYn48mewpyQv5ByYI/edit#slide=id.g10d42026b8_2_0)
 