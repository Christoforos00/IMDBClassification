# IMDBClassification
In this project we implement the SGD Logistic Regression and Naive Bayes algorithms, and use them to classify IMDB comments.


Our dataset contains 50000 .txt files of IDMB comments, which are labeled as positive or negative. The main project is executed in mainNotebook.ipynb .
Initially we preprocess the files so that only alphanumerics and spaces are kept. Then, we create a dictionary out of the n most frequent words that appear in our dataset, where the m first are skipped.
As we mention below we decided n=2000 and m=0 , which means that the dictionary contains the 2000 most frequent words.
We convert each comment to a vector(1x2000) which represents the existance of each one of the dictionary's words in the current comment.
After converting all the comments ,we tune the hyperparameters (the parameters of our algorithms but also the n,m values) using the CV data .
Finally for each algorithm we calculate the learning curves and accuracy (>80%).

From Logistic regression we get the following diagrams:

![1](https://user-images.githubusercontent.com/50478180/109492511-c5aa8500-7a93-11eb-8c38-321a93b6552b.png)
![2](https://user-images.githubusercontent.com/50478180/109492596-e246bd00-7a93-11eb-89cc-706de44ad812.png)


From Naive Bayes we get the following diagrams:

![3](https://user-images.githubusercontent.com/50478180/109492664-fdb1c800-7a93-11eb-9528-00f330b97356.png)
![4](https://user-images.githubusercontent.com/50478180/109492669-ff7b8b80-7a93-11eb-85c2-538fdd16276b.png)


The code files are
• logReg.py, where the functions of the Logistic Regression algorithm are implemented

• naiveBayes.py, where the functions of the Naive Bayes algorithm are implemented

• dataUtils.py, which contains functions for editing and converting text

• mainNotebook.ipynb, where the main part of the work is located

The dataset is located at https://ai.stanford.edu/~amaas/data/sentiment/ .
