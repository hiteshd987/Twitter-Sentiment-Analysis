from sklearn.linear_model import LogisticRegression

def calc_log_reg(X_train, X_test, y_train, tfidf_vec): 
    X_train_logistic_reg = tfidf_vec.fit_transform(X_train)
    X_test_logistic_reg = tfidf_vec.transform(X_test)
    
    logistic_reg_model = LogisticRegression(max_iter=1000)
    logistic_reg_model.fit(X_train_logistic_reg, y_train)
    
    logisticreg_pred = logistic_reg_model.predict(X_test_logistic_reg)
    return logisticreg_pred