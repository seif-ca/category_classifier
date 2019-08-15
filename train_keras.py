# in case this is run outside of conda environment with python2

from sklearn import metrics

import mlflow
import sys
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import mlflow.keras
from hyperopt import fmin, hp, tpe, STATUS_OK,space_eval
from hyperopt.pyll.base import scope
from hyperopt import Trials
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils

mlflow.keras.autolog()
#spark.conf.set('spark.databricks.mlflow.trackHyperopt.enabled', 'true')

#mlflow.set_tracking_uri('databricks')
#mlflow.set_experiment('/Users/quentin.ambard@databricks.com/MLFlow/02-Classification-products')


local_path = "data/products.csv"

def get_cloud_data():
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    return spark.sql("select * from quentin.products").toPandas()

def get_local_data():
    return pd.read_csv(local_path)

def prepare_dataset(df):
  le = LabelEncoder()
  le.fit(df["category"])
  x_train, x_test, y_train, y_test = train_test_split(df["description"], le.transform(df["category"]), test_size=0.33, random_state=42)
  return x_train, x_test, y_train, y_test

class RunKeras:

    def __init__(self, train_desc, test_desc, train_cat, test_cat):
        self.max_words = 1000
        tokenize = text.Tokenizer(num_words=self.max_words, char_level=False)
        tokenize.fit_on_texts(train_desc)  # only fit on train
        self.x_train = tokenize.texts_to_matrix(train_desc)
        self.x_test = tokenize.texts_to_matrix(test_desc)
        self.num_classes = np.max(train_cat) + 1
        self.y_train = utils.to_categorical(train_cat, self.num_classes)
        self.y_test = test_cat

    def train_model(self, params):
        batch_size = params['batch_size']
        epochs = params['epochs']

        # Build the model
        model = Sequential()
        model.add(Dense(512, input_shape=(self.max_words,)))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout']))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=params['optimizer'],
                      metrics=['accuracy'])

        model.fit(self.x_train, self.y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_split=0.1)
        predictions = model.predict(self.x_test)
        y_pred = np.argmax(predictions, axis=1)
        f1 = metrics.f1_score(self.y_test, y_pred, average='micro')
        return model, f1

    def keras_model(self, params):
        with mlflow.start_run(run_name="keras_trial", nested=True):
            model, f1 = self.train_model(params)
        return {'loss': -f1, 'status': STATUS_OK}  # 'attachments': {'model': model}}


def get_search_space():
    return {'batch_size': scope.int(hp.uniform('batch_size', 30, 40)),
            'epochs': hp.choice('epochs', [2, 5]),
            'activation': hp.choice('activation', ['relu', 'elu']),
            'optimizer': hp.choice('optimizer', ['adam', 'sgd']),
            'dropout': hp.uniform('dropout', 0, 1)}


def train_keras_model(train_desc, test_desc, train_cat, test_cat, distributed = False,shop="all"):
    with mlflow.start_run(run_name="keras", nested=True):
        if(distributed):
            from hyperopt import SparkTrials
            trials = SparkTrials()
        else:
            trials = Trials()
        run_keras = RunKeras(train_desc, test_desc, train_cat, test_cat)
        argmin = fmin(run_keras.keras_model, get_search_space(), algo=tpe.suggest, max_evals=10, show_progressbar=True,
                      trials=trials)
        best_params = space_eval(get_search_space(), argmin)
        best_model, f1 = run_keras.train_model(best_params)
        #mlflow.keras.log_model(best_model, 'model')
        mlflow.log_metric("f1", f1)
        #mlflow.log_metric("delta_version", delta_version)
        mlflow.set_tag("shop", shop)
        mlflow.set_tag("model", "keras_classifier")
        return argmin

def run(remote):
    df = get_local_data()
    x_train, x_test, y_train, y_test = prepare_dataset(df[:1000])
    train_keras_model(x_train, x_test, y_train, y_test,distributed=remote)

if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    remote = (int(sys.argv[1])>0)
    run(remote)
