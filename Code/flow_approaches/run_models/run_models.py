from classifiers.cnn_classifier import cnn_classifier
from classifiers.knn_classifier import knn_classifier
from classifiers.mlp_classifier import mlp_classifier
from classifiers.naive_bayes_classifier import naive_bayes_classifier
from classifiers.random_forest_classifier import random_forest_classifier
from classifiers.svm_classifier import svm_classifier


def run_models(
    data_train, categories_train, data_dev, categories_dev, data_test, categories_test
):
    results = {}

    print("\nRunning MLP Classifier")
    mlp_results = mlp_classifier(
        data_train,
        categories_train,
        data_dev,
        categories_dev,
        data_test,
        categories_test,
    )
    results["MLP"] = mlp_results

    print("\nRunning CNN Classifier")
    cnn_results = cnn_classifier(
        data_train,
        categories_train,
        data_dev,
        categories_dev,
        data_test,
        categories_test,
    )
    results["CNN"] = cnn_results

    print("\nRunning Random Forest Classifier")
    random_forest_results = random_forest_classifier(
        data_train,
        categories_train,
        data_dev,
        categories_dev,
        data_test,
        categories_test,
    )
    results["Random Forests"] = random_forest_results

    print("\nRunning Naive Bayes Classifier")
    naive_bayes_results = naive_bayes_classifier(
        data_train,
        categories_train,
        data_dev,
        categories_dev,
        data_test,
        categories_test,
    )
    results["Naive Bayes"] = naive_bayes_results

    print("\nRunning KNN Classifier")
    knn_results = knn_classifier(
        data_train,
        categories_train,
        data_dev,
        categories_dev,
        data_test,
        categories_test,
        5,
    )
    results["KNN"] = knn_results

    # SVM takes too long to be viable
    """
    print("\nRunning SVM Classifier")
    svm_results = svm_classifier(
        data_train, categories_train, data_test, categories_test
    )
    results["SVM"] = svm_results
    """

    return results
