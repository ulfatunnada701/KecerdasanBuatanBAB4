
# Exercise 4.2 Multiple Layer Perceptron
from sklearn.neural_network import MLPClassifier

# Data training dengan 3 input
X = [[0., 0., 0.], [1., 1., 1.]]
y = [0, 1]

# Membuat model MLP
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)

# Melatih model
clf.fit(X, y)

# Menguji model dengan 3 input
print(clf.predict([[2., 2., 2.], [-1., -1., -2.]]))

# Menampilkan struktur bobot
print([coef.shape for coef in clf.coefs_])
