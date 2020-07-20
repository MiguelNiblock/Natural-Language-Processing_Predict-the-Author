# NLP- Predict the Author

This procedure compares various techniques used in NLP such as Latent Semantic Analysis, Bag of Words and TFIDF; to predict the author for a given article.

![banner](Deliverables/output_121_0.png)

## Summary

- **[Notebook](https://miguelniblock.github.io/Natural-Language-Processing_Predict-the-Author/docs/index.html) with procedure-** This single-notebook Python procedure has its own TOC and includes headings like "Introduction to Dataset", "Exploratory Data Analysis", "Supervised Feature Generation", "Unsupervised Feature Generation", and "Choosing  Model".
    - Also as [PDF](https://miguelniblock.github.io/Natural-Language-Processing_Predict-the-Author/Deliverables/NLP-Predict_the_Author.pdf).

## Context

NLP techniques like **Bag-of-Words** and **Latent Semantic Analysis** can be useful to turn a raw pool of text into a tabular format that standard machine learning algorithms can understand. 

If we take a given text as training data, and take a group of author names as prediction target, we can build a supervised classification model that learns to identify an author's style of writing. 

The table below shows the final results of my project. From top to bottom, are the best-performing algorithms as well as an indicator for which technique was used for feature-engineering: **BOW**(*Bag-of-Words*) VS **LSA**(*Latent Semantic Analysis*)

|      |                  Algorithm | n_train | Features | Mutual_Info | Test_Accuracy |
| ---: | -------------------------: | ------: | -------: | ----------: | ------------: |
|    4 |         LogisticRegression |     380 |      BOW |    0.837992 |      0.841667 |
|    5 |     RandomForestClassifier |     380 |      BOW |    0.762783 |      0.791667 |
|    6 | GradientBoostingClassifier |     380 |      BOW |    0.744608 |         0.775 |
|   18 |         LogisticRegression |     760 |      LSA |    0.687342 |      0.758333 |
|   11 |         LogisticRegression |     380 |      LSA |    0.724234 |           0.7 |
|   12 |     RandomForestClassifier |     380 |      LSA |    0.705238 |      0.683333 |
|   19 |     RandomForestClassifier |     760 |      LSA |    0.590879 |      0.645833 |
|   13 | GradientBoostingClassifier |     380 |      LSA |    0.607244 |      0.633333 |
|   20 | GradientBoostingClassifier |     760 |      LSA |    0.512496 |        0.5625 |
|    0 |                     KMeans |     500 |      BOW |    0.428272 |           NaN |

