from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

def calc_svm(X_train, X_test, y_train):
    train_vec_SVM = vectorizer.fit_transform(X_train)
    test_vec_SVM = vectorizer.transform(X_test)
    
    clf_svm = svm.SVC(kernel='linear')
    clf_svm.fit(train_vec_SVM, y_train)
    
    pred_svm = clf_svm.predict(test_vec_SVM)
    return pred_svm