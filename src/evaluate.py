from  sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(y_test, y_preds):

    conf_matrix=confusion_matrix(y_test, y_preds)
    print("Confusion Matrix:"+"\n", conf_matrix)

    class_report=classification_report(y_test, y_preds)
    print("Classification Report:" + "\n", class_report)

