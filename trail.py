# importing the module 
import numpy as np
import tensorflow as tf
import keras_tuner
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import GridSearch, HyperModel, Objective, HyperParameters
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reduce the size of the dataset for faster training
n = 8000
x_train = x_train[:n]
y_train = y_train[:n]
# num_test_samples = 2000
# x_test = x_test[:num_test_samples]
# y_test = y_test[:num_test_samples]

# Normalize the images
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure the images have the right shape
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode the labels
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

best_model=None
history={}

class CustomHyperModel(HyperModel):
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
                layers.Dense(num_classes, activation="softmax"),
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

best_model=None
history=None

tuner = GridSearch(
    CustomHyperModel(),
    objective=Objective("val_loss", direction="min"),
    max_trials=30,
    executions_per_trial=3,
    directory="grid_search_results_fulldata2",
    project_name="mnist_grid_search_fulldata_tuning2",
    seed=42,
)

tuner.search_space_summary()

tuner.search(
    x_train,
    y_train,
    epochs=5,
    validation_split=0.1,
    verbose=2,
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters:")
print(best_hp.get_config())


# Build the best model with the best hyperparameters
best_model = tuner.hypermodel.build(best_hp)
best_batch_size = best_hp.get("batch_size")

# Train the best model
history = best_model.fit(
    x_train,
    y_train,
    batch_size=best_batch_size,
    epochs=5,
    validation_split=0.1,
)

# Evaluate the best model on the test data
test_loss, test_accuracy = best_model.evaluate(x_test, y_test)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)

def plot_trial_history(val_losses, val_accuracies, test_losses, test_accuracies):
    num_trials = len(val_losses)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_trials + 1), val_losses, label="Validation loss")
    plt.plot(range(1, num_trials + 1), test_losses, label="Test loss")
    plt.xlabel("Trial number")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss vs. trial number")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_trials + 1), val_accuracies, label="Validation accuracy")
    plt.plot(range(1, num_trials + 1), test_accuracies, label="Test accuracy")
    plt.xlabel("Trial number")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy vs. trial number")

    plt.show()

val_losses = []
val_accuracies = []
test_losses = []
test_accuracies = []

for trial in tuner.oracle.trials.values():
    val_loss = trial.metrics.get_last_value("val_loss")
    val_accuracy = trial.metrics.get_last_value("val_accuracy")
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    model = tuner.hypermodel.build(trial.hyperparameters)
    model.fit(
        x_train,
        y_train,
        batch_size=trial.hyperparameters.get("batch_size"),
        epochs=5,
        validation_split=0.1,
        verbose=0,
    )
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

# Call the plot_trial_history() function after the search is done
plot_trial_history(val_losses, val_accuracies, test_losses, test_accuracies)

# Plot the learning curves
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

