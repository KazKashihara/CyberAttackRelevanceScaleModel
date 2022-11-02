from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn import metrics
import scikitplot as skplt

import pandas as pd

if __name__ == '__main__':
    from sklearn.feature_extraction.text import TfidfVectorizer


    train = pd.read_csv("./Data/trainData.csv", encoding="utf-8")
    test = pd.read_csv("./Data/testData.csv", encoding="utf-8")

    train_X = train["post"].tolist()
    test_X = test["post"].tolist()

    train_y = train["score"].tolist()
    test_y = test["score"].tolist()

    test_E = test["Exploit"].tolist()

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english')

    features = tfidf.fit_transform(train_X).toarray()
    labels = train_y
    features.shape

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_X)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, train_y)

    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []

    print(len(features))
    print(len(labels))
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

    print(cv_df.groupby('model_name').accuracy.mean())
    labelInfo = ["0", "1", "2", "3"]

    # Read data
    df = pd.read_csv("./Data/Data.csv")
    labels2 = df['Score'].tolist()
    posts = df['Post Content'].tolist()

    datasets = pd.DataFrame(columns=["post", "score"])
    for i in range(len(labels)):
        post = posts[i]
        score = labels[i]
        s = pd.Series([post, score], index=datasets.columns)
        datasets = datasets.append(s, ignore_index=True)

    # Randomise the data
    datasets = datasets.sample(frac=1).reset_index(drop=True)
    datasets.head()
    features2 = tfidf.fit_transform(posts).toarray()

    trainf_X, testf_X, train_y, test_y, indices_train, indices_test = train_test_split(features2, labels2, df.index,
                                                                                     test_size=0.2, random_state=0)
    #train, test = train_test_split(datasets, test_size=0.2)

    #trainf_X = tfidf.fit_transform(train_X).toarray()
    #testf_X = tfidf.fit_transform(test_X).toarray()
    weights = {0:1.0, 1:8.25, 2:27.76, 3:82.56}

    print("RandomForest")
    model = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0, class_weight=weights)
    model.fit(trainf_X, train_y)
    y_pred = model.predict(testf_X)

    conf_mat = confusion_matrix(test_y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_mat, cmap="Blues", annot=True, fmt='d',
                xticklabels=labelInfo, yticklabels=labelInfo,linewidths=1, linecolor='black')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    outfile = "./RandomForest-Performance.png"
    plt.savefig(outfile)

    skplt.metrics.plot_confusion_matrix(
        test_y,
        y_pred,
        figsize=(12, 12), x_tick_rotation=90).figure.savefig('./RandomForest-Result.png')

    print(metrics.classification_report(test_y, y_pred,
                                        target_names=labelInfo))
    print()

    print("LinearSVC")
    model = LinearSVC(class_weight=weights)
    model.fit(trainf_X, train_y)
    y_pred = model.predict(testf_X)

    conf_mat = confusion_matrix(test_y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_mat, cmap="Blues", annot=True, fmt='d',
                xticklabels=labelInfo, yticklabels=labelInfo,linewidths=1, linecolor='black')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    outfile = "./LinearSVC-Performance.png"
    plt.savefig(outfile)

    skplt.metrics.plot_confusion_matrix(
        test_y,
        y_pred,
        figsize=(12, 12), x_tick_rotation=90).figure.savefig('./LinearSVC-Result.png')

    print(metrics.classification_report(test_y, y_pred,
                                        target_names=labelInfo))
    print()

    print("MultinomialNB")
    model = MultinomialNB()
    model.fit(trainf_X, train_y)
    y_pred = model.predict(testf_X)

    conf_mat = confusion_matrix(test_y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_mat, cmap="Blues", annot=True, fmt='d',
                xticklabels=labelInfo, yticklabels=labelInfo,linewidths=1, linecolor='black')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    outfile = "./MultinomialNB-Performance.png"
    plt.savefig(outfile)

    skplt.metrics.plot_confusion_matrix(
        test_y,
        y_pred,
        figsize=(12, 12), x_tick_rotation=90).figure.savefig('./MultinomialNB-Result.png')

    print(metrics.classification_report(test_y, y_pred,
                                        target_names=labelInfo))
    print()

    print("LogisticRegression")
    modelLR = LogisticRegression(random_state=0,class_weight=weights)
    modelLR.fit(trainf_X, train_y)
    y_pred = modelLR.predict(testf_X)

    conf_mat = confusion_matrix(test_y, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_mat, cmap="Blues", annot=True, fmt='d',
                xticklabels=labelInfo, yticklabels=labelInfo,linewidths=1, linecolor='black')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    outfile = "./LogisticRegression-Performance.png"
    plt.savefig(outfile)

    skplt.metrics.plot_confusion_matrix(
        test_y,
        y_pred,
        figsize=(12, 12), x_tick_rotation=90).figure.savefig('./LogisticRegression-Result.png')

    print(metrics.classification_report(test_y, y_pred,
                                        target_names=labelInfo))
    print()

    #df1 = pd.read_csv("./Data/Evaluation-Posts.csv")
    #postList = df1['Post Content'].tolist()
    resultList = []
    text_features = tfidf.transform(test_X)
    predictions = modelLR.predict(text_features)

    for text, predicted, score, exploit in zip(test_X, predictions, test_y, test_E):
        pairList = []
        pairList.append(text)
        pairList.append(predicted)
        pairList.append(score)
        pairList.append(exploit)
        resultList.append(pairList)

    postSaveFile = "./Data/LogisticRegression-Evaluation-Result.csv"
    S1 = pd.DataFrame(resultList, columns=['Post Content', 'Predicted Score', 'Score', 'Exploit'])
    S1.to_csv(postSaveFile)

    dfEx = pd.read_csv("./Data/ExploitedPosts.csv")
    exploitedPosts = dfEx["Post Content"].tolist()
    text_features = tfidf.transform(exploitedPosts)
    predictions = modelLR.predict(text_features)
    resultList = []
    for post, predicted in zip(exploitedPosts, predictions):
        pairList = []
        pairList.append(post)
        pairList.append(predicted)
        resultList.append(pairList)

    postSaveFile = "./Data/LR-Exploit-Result.csv"
    S1 = pd.DataFrame(resultList, columns=['Post Content', 'Predicted Score'])
    S1.to_csv(postSaveFile)

