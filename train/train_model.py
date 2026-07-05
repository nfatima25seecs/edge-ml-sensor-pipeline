import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

np.random.seed(0)

def make_activity(center, n=100):
    return np.random.randn(n, 12) + center

walk = make_activity(0.0)
sit  = make_activity(3.0)
run  = make_activity(6.0)

X = np.vstack([walk, sit, run])
y = np.array([0]*100 + [1]*100 + [2]*100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

model = MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=1)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("Accuracy on unseen data:", round(acc * 100, 1), "%")
print("Activities: ['walk', 'sit', 'run']")
