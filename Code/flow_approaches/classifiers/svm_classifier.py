from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as score


def svm_classifier(
    data_train, categories_train, data_dev, categories_dev, data_test, categories_test
):
    svc = svm.SVC()
    svc.fit(data_train, categories_train)
    categories_pred = svc.predict(data_dev)
    accuracy_dev = accuracy_score(categories_dev, categories_pred)
    macro_precision_dev, macro_recall_dev, macro_fscore_dev, macro_support_dev = score(
        categories_dev, categories_pred, average="macro"
    )
    print(f"Dev Accuracy: {accuracy_dev}")
    print(f"Dev Macro-Precision: {macro_precision_dev}")
    print(f"Dev Macro-Recall: {macro_recall_dev}")
    print(f"Dev Macro-F-Score: {macro_fscore_dev}")
    print(classification_report(categories_test, categories_pred, zero_division=1))

    categories_pred = svm.predict(data_test)
    accuracy_test = accuracy_score(categories_test, categories_pred)
    macro_precision_test, macro_recall_test, macro_fscore_test, macro_support_test = (
        score(categories_test, categories_pred, average="macro")
    )
    print(f"Test Accuracy: {accuracy_test}")
    print(f"Test Macro-Precision: {macro_precision_test}")
    print(f"Test Macro-Recall: {macro_recall_test}")
    print(f"Test Macro-F-Score: {macro_fscore_test}")
    print(classification_report(categories_test, categories_pred, zero_division=1))
    return {
        "Dev Accuracy": accuracy_dev,
        "Dev Macro-Precision": macro_precision_dev,
        "Dev Macro-Recall": macro_recall_dev,
        "Dev Macro-F-Score": macro_fscore_dev,
        "Test Accuracy": accuracy_test,
        "Test Macro-Precision": macro_precision_test,
        "Test Macro-Recall": macro_recall_test,
        "Test Macro-F-Score": macro_fscore_test,
    }
