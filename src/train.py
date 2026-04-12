import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from data_preprocessing import load_data, clean_data, split_features
from evaluate import evaluate_model

df=load_data("WA_Fn-UseC_-Telco-Customer-Churn.csv")    
df=clean_data(df)

X,y=split_features(df)

X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=42, test_size=0.2)

scale=StandardScaler()
X_train=scale.fit_transform(X_train)
X_test=scale.transform(X_test)

models={
    "Random Forest Classifier": {
        "model": RandomForestClassifier(), 
        "params":{"n_estimators":[100,200], "max_depth":[10,20]}
    },
    "SVC": {
        "model": SVC(probability=True),
        "params":{"C":[0.1,1], "kernel":["linear", "rbf"]}
    },
    "Logistic Regression": {
        "model": LogisticRegression(), 
        "params":{"C":[0.1,1,10]}
    }
}

results=[]

for name, model in models.items():
    print("-"*50)
    print(f"Training {name}")
    print("-"*50)
    grid=GridSearchCV(cv=5, scoring="f1", estimator=models[name]["model"], param_grid=models[name]["params"])
    grid.fit(X_train, y_train)

    best_model=grid.best_estimator_
    y_preds=best_model.predict(X_test)
    y_probs=best_model.predict_proba(X_test)[:,1]

    f1=f1_score(y_test,y_preds)
    roc=roc_auc_score(y_test, y_probs)
    results.append((f1, roc, name))

    print("Best Params:\n", grid.best_params_)
    print("F1 Score:\n", f1)
    print("ROC AUC Score:\n", roc)
    
    metrics = evaluate_model(y_test, y_preds)
    print("Confusion Matrix:\n", metrics["confusion_matrix"])
    print("Classification Report:\n", metrics["classification_report"])
