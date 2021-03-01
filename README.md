# IMDBClassification
In this project we implement the SGD Logistic Regression and Naive Bayes algorithms to classify IMDB comments.


Our dataset contains 50000 .txt files of IDMB comments, which are labeled as positive or negative. The main project is executed in mainNotebook.ipynb .
Initially we preprocess the files so that only alphanumerics and spaces are kept. Then, we create a dictionary out of the n most frequent words that appear in our dataset, where the m first are skipped.
In this case we decided n=2000 and m=0 , which means that the dictionary contains the 2000 most frequent words.

want to convert each comment to a vector which represents the existance of a word in a predefined dictionary of words.

The code files are
• logReg.py, where the functions of the Logistic Regression algorithm are implemented
• naiveBayes.py, where the functions of the Naive Bayes algorithm are implemented
• dataUtils.py, which contains functions for editing and converting text
• mainNotebook.ipynb, where the main part of the work is located

The dataset is located here https://ai.stanford.edu/~amaas/data/sentiment/ .
