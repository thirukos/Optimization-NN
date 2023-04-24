import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import GridSearch, RandomSearch, HyperModel, Objective, HyperParameters
import matplotlib.pyplot as plt
from keras.datasets import mnist
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, n=8000):
        self.n = n

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data() # training data size 60000, test data size 10000
        x_train = x_train[:self.n]
        y_train = y_train[:self.n]

        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        num_classes = 10
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        return x_train, y_train, x_test, y_test

class MyHyperModel(HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        hp_optimizer = hp.Choice("optimizer", values=["adam", "sgd"])
        hp_batch_size = hp.Int("batch_size", min_value=32, max_value=256, step=32)

        if hp_optimizer == "adam":
            optimizer = keras.optimizers.Adam(
                learning_rate=hp_learning_rate,
                beta_1=hp.Float("beta1", min_value=0.5, max_value=0.999, step=0.1),
                beta_2=hp.Float("beta2", min_value=0.5, max_value=0.999, step=0.1),
                epsilon=1e-7,
            )
        else:
            optimizer = keras.optimizers.SGD(
                learning_rate=hp_learning_rate,
                momentum=hp.Float("momentum", min_value=0, max_value=0.99, step=0.1),
                nesterov=True,
            )
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

class NelderMeadHyperModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, params):
        # Map the input params array to the corresponding hyperparameters
        dropout_rate, learning_rate, batch_size = params
        batch_size = int(batch_size)

        model = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(dropout_rate),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,  # Fixed values for beta1 and beta2
            beta_2=0.999,
            epsilon=1e-7,
        )

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
    
class TunerWrapper:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def grid_search(self, hypermodel, dir, pname):
        tuner = GridSearch(
            hypermodel,
            objective=Objective("val_loss", direction="min"),
            max_trials=30,
            executions_per_trial=3,
            directory=dir,
            project_name=pname,
            seed=42,
        )
        return self._run_keras_tuner(tuner)

    def random_search(self, hypermodel, dir, pname):
        tuner = RandomSearch(
            hypermodel,
            objective=Objective("val_loss", direction="min"),
            max_trials=30,
            executions_per_trial=3,
            directory=dir,
            project_name=pname,
            seed=42,
        )
        return self._run_keras_tuner(tuner)

    def nelder_mead_search(self, hypermodel):
        return self._run_nelder_mead(hypermodel)

    def _run_keras_tuner(self, tuner):
        tuner.search_space_summary()
        tuner.search(
            self.x_train,
            self.y_train,
            epochs=5,
            validation_split=0.1,
            verbose=2,
        )

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters:")
        print(best_hp.get_config())

        best_model = tuner.hypermodel.build(best_hp)
        best_batch_size = best_hp.get("batch_size")
        history = best_model.fit(
            self.x_train,
            self.y_train,
            batch_size=best_batch_size,
            epochs=5,
            validation_split=0.1,
        )

        test_loss, test_accuracy = best_model.evaluate(self.x_test, self.y_test)
        print("Test loss:", test_loss)
        print("Test accuracy:", test_accuracy)

        return best_model, history, best_hp

    def _run_nelder_mead(self, hypermodel):
        def objective_function(params):
            model = hypermodel.build(params)
            history = model.fit(
                self.x_train,
                self.y_train,
                batch_size=int(params[2]),
                epochs=5,
                validation_split=0.1,
                verbose=0,
            )
            val_loss = history.history["val_loss"][-1]
            return val_loss

        x0 = np.array([0.1, 1e-3, 32])  # Replace with appropriate initial values

        result = minimize(objective_function, x0, method="Nelder-Mead")

        best_params = result.x
        best_model = hypermodel.build(best_params)
        best_batch_size = int(best_params[2])
        history = best_model.fit(
            self.x_train,
            self.y_train,
            batch_size=best_batch_size,
            epochs=5,
            validation_split=0.1,
        )

        test_loss, test_accuracy = best_model.evaluate(self.x_test, self.y_test)
        print("Test loss:", test_loss)
        print("Test accuracy:", test_accuracy)
        return best_model, history, best_params

class PlotResults:
    def __init__(self, results):
        self.results = results

    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        for algorithm, history in self.results.items():
            plt.plot(history["loss"], label=f"{algorithm} Training loss")
            plt.plot(history["val_loss"], label=f"{algorithm} Validation loss")
        
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss vs. Epochs")
        plt.show()

    def plot_accuracy(self):
        plt.figure(figsize=(10, 5))
        for algorithm, history in self.results.items():
            plt.plot(history["accuracy"], label=f"{algorithm} Training accuracy")
            plt.plot(history["val_accuracy"], label=f"{algorithm} Validation accuracy")
        
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Accuracy vs. Epochs")
        plt.show()