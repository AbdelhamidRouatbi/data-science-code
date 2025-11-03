# --- imports : make sure you run "pip install wandb" and "wandb login" before --- #
import wandb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- Initialize wandb ---
wandb.init(project="milestone_2", name="iris knn test", config={
    "n_neighbors": 5,
    "weights": "uniform",
    "test_size": 0.2,
    "some_param": 3, # you can specify hyperparams here
    "random_state": 42
})

config = wandb.config

# --- Load data ---#
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.test_size, random_state=config.random_state
)

# --- Train model ---#
model = KNeighborsClassifier(n_neighbors=config.n_neighbors, weights=config.weights)
model.fit(X_train, y_train)

# --- Evaluate ---#
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- Log metrics ---#
wandb.log({"accuracy": acc})

print(f"Accuracy: {acc:.3f}")
wandb.finish()

