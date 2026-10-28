
from numpy import exp, array, random, dot

# Set up parameters
inputs = array([[0, 0], [1, 1], [1, 0], [0, 1]])
outputs = array([[0, 1, 1, 1]]).T

random.seed(1)
weights = 2 * random.random((3, 1)) - 1

# Menambahkan bias ke setiap input
inputs = array([[0, 0, 1],
                [1, 1, 1],
                [1, 0, 1],
                [0, 1, 1]])

# Define a Single Neuron function
def neuron(inputs, weights):
    output = 1 / (1 + exp(-(dot(inputs, weights))))
    return output

# Train the Neuron
for iteration in range(50000):
    output = neuron(inputs, weights)
    weights += dot(inputs.T, (outputs - output) *
                   output * (1 - output))

# Test the Neuron (Logical OR)
print("Hasil Pengujian Logika OR:")
for a in [0, 1]:
    for b in [0, 1]:
        x = array([a, b, 1])
        prediction = neuron(x, weights)[0]
        result = int(prediction >= 0.5)
        print(f"{a} OR {b} = {result} (Prediksi: {prediction:.4f})")
