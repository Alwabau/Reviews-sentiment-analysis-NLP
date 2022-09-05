# Sentiment Analysis on Movies Reviews

# Sentiment Analysis: Large Movie Review Dataset

We are going to use the data from AI Stanford Dataset (https://ai.stanford.edu/~amaas/data/sentiment/). This is a dataset for binary sentiment classification, where the possible output labels are: "positive" and "negative". Which indicates, if the review of a movie speaks positively or negatively. 

### These are the objectives of the project:

* Read data that is not in a traditional format.
* Put together a set of preprocessing functions that we can use later on any NLP or related problems.
* Vectorize the data in order to apply a machine learning model to it: using TF-IDF.
* TF-IDF (and also BoW) is a classic way to vectorize text, but currently we have some more complex ways with better performance, for this we are going to train our own word embedding and use it as a vectorization source for our data.
* Train a sentiment analysis model that allows us to detect positive and negative opinions in movie reviews.

## Install

You can use `Docker` to easily install all the needed packages and libraries:

```bash
$ docker build -t sprint6_project -f ./Dockerfile .
```

### Run Docker

```bash
$ docker run --rm -it -p 8888:8888 -v $(pwd):/home/app/src sprint6_project bash
```

## Run Project

It doesn't matter if you are inside or outside a Docker container, in order to execute the project you need to launch a Jupyter notebook server running:

```bash
$ jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
```
