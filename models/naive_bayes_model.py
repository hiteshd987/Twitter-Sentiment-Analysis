from sklearn.naive_bayes import MultinomialNB

def calc_naive_bayes(X_train, X_test, y_train, tfidf_vec):
    
    X_train_vec = tfidf_vec.fit_transform(X_train)
    X_test_vec = tfidf_vec.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    pred_nb = clf.predict(X_test_vec)
    return pred_nb