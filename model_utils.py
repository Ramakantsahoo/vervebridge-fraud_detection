from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

def split_data(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)

def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5, p=1)
    knn.fit(X_train, y_train)
    return knn

def train_rf(X_train, y_train):
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, verbose=1, class_weight="balanced")
    rf_clf.fit(X_train, y_train)
    return rf_clf

def train_xgb(X_train, y_train):
    XGBoost_CLF = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=400, objective="binary:hinge",
                                    booster='gbtree', n_jobs=-1, random_state=42, verbosity=1)
    XGBoost_CLF.fit(X_train, y_train)
    return XGBoost_CLF

def create_ensemble(knn, rf_clf, XGBoost_CLF):
    estimators = [("KNN", knn), ("rf", rf_clf), ("xgb", XGBoost_CLF)]
    ens = VotingClassifier(estimators=estimators, voting="soft", weights=[1, 4, 1])
    return ens

def train_ensemble(ens, X_train, y_train):
    ens.fit(X_train, y_train)
    return ens

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report: \n", classification_report(y_test, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    return y_pred
