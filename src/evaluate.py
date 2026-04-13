from  sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model(y_test, y_preds):

    conf_matrix=confusion_matrix(y_test, y_preds)
    class_report=classification_report(y_test, y_preds)
    
    return {"confusion_matrix": conf_matrix, "classification_report": class_report}

