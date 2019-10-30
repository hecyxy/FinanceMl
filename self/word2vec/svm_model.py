import pandas as pd
from sklearn.svm import SVC
from sklearn.externals import joblib

def svm_train(train_vec,y_train,test_vec,y_test):
    clf = SVC(kernel='rbf',verbose=True)
    clf.fit(train_vec,y_train)
    joblib.dump(clf,'./self/word2vec/svm_model.pkl',compress=3)
    print(clf.score(test_vec,y_test))

