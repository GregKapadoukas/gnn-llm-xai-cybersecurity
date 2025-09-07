import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from tensorflow import keras


def mlp_classifier(
    data_train, categories_train, data_dev, categories_dev, data_test, categories_test
):
    categories_train = pd.get_dummies(categories_train)
    categories_test = pd.get_dummies(categories_test)
    categories_dev = pd.get_dummies(categories_dev)
    categories_list = categories_train.columns
    model = keras.Sequential(
        [
            keras.Input(data_train.shape[1]),
            layers.Dense(60, activation="relu"),
            layers.Dense(30, activation="relu"),
            layers.Dense(10, activation="relu"),
            layers.Dense(categories_train.shape[1], activation="softmax"),
        ]
    )

    if categories_train.shape[1] == 2:
        model.compile(
            loss=keras.losses.BinaryCrossentropy(),
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001, beta_1=0.6, beta_2=0.999
            ),
            metrics=["binary_crossentropy", "categorical_accuracy"],
        )

    else:
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=keras.optimizers.Adam(
                learning_rate=0.001, beta_1=0.6, beta_2=0.999
            ),
            metrics=["categorical_crossentropy", "categorical_accuracy"],
        )

    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

    model.fit(
        data_train,
        categories_train,
        batch_size=32,
        epochs=20,
        validation_split=0.2,
        callbacks=[callback],
        verbose=2,
    )

    categories_pred = model.predict(data_dev)
    categories_pred = [categories_list[np.argmax(p)] for p in categories_pred]
    categories_dev = categories_dev.to_numpy()
    categories_dev = [categories_list[np.argmax(p)] for p in categories_dev]
    accuracy_dev = accuracy_score(categories_dev, categories_pred)
    macro_precision_dev, macro_recall_dev, macro_fscore_dev, macro_support_dev = score(
        categories_dev, categories_pred, average="macro"
    )
    print(f"Dev Accuracy: {accuracy_dev}")
    print(f"Dev Macro-Precision: {macro_precision_dev}")
    print(f"Dev Macro-Recall: {macro_recall_dev}")
    print(f"Dev Macro-F-Score: {macro_fscore_dev}")
    print(classification_report(categories_dev, categories_pred, zero_division=1))

    categories_pred = model.predict(data_test)
    categories_pred = [categories_list[np.argmax(p)] for p in categories_pred]
    categories_test = categories_test.to_numpy()
    categories_test = [categories_list[np.argmax(p)] for p in categories_test]
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
