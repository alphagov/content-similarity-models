# content-similarity-models
> Experiments and use of content similarity models

A set of experiments and models to generate embeddings for content items on GOV.UK, so far:
1. [doc2vec](https://radimrehurek.com/gensim/models/doc2vec.html)
2. [Google's universal sentence encoder](https://tfhub.dev/google/universal-sentence-encoder/2)

# Installing / Getting started
When following this guidance, code is executed in the terminal unless specified otherwise.  

Clone this repo using:  
 
`git clone git@github.com:alphagov/content-similarity-models.git`  

in your terminal.  

## Where to get the data

These models either use clean.content.csv.gz or labelled.csv.gz which are produced in the 
dataprep scripts in [alphagov/govuk-taxonomy-supervised-learning](https://github.com/alphagov/govuk-taxonomy-supervised-learning)
  

## Python version
These experiments were conducted in python version [Python 3.6.4](https://www.python.org/downloads/release/python-360/).  

## Virtual environment
Create a new python 3.6.4 virtual environment using your favourite virtual environment 
manager (you may need `pyenv` to specify python version; which you can get using `pip install pyenv`). 

If new to python, an easy way to do this is using the PyCharm community edition and opening this repo as a project. 
You can then specify what python interpreter to use (as 
explained [here](https://stackoverflow.com/questions/41129504/pycharm-with-pyenv)).  


## Using pip to install necessary packages
Then install required python packages:  

`pip install -r requirements.txt`  

## How to visualise the embeddings in tensorboard

After saving checkpoints to specified log directory e.g., universal_embeddings:

`tensorboard --logdir=universal_embeddings`

then go to `localhost:6006` in your browser. They can take a long time to render.

You can then play around with the t-SNE and PCA algorithms for visualisation and colour the pages by 
any metadata items.

# Contributing


## License

