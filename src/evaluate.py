from  sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from train import X_test, y_test, y_preds

def evaluate_model(y_test, y_preds):

    confusion_matrix=confusion_matrix(y_test, y_preds)
    print("Confusion Matrix:"+"\n", confusion_matrix)

    classification_report=classification_report(y_test, y_preds)
    print("Classification Report:" + "\n", classification_report)

