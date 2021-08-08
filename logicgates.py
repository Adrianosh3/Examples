from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# GATE PERCEPTRON PROJECT
data = [[0, 0],[0, 1],[1, 0],[1, 1]]
#labels = [0, 1, 1, 0] # XOR Gate not linear seperatable
labels = [0, 1, 1, 1] # OR Gate 100%
#labels = [0, 0, 0, 1] # AND Gate 100%
plt.scatter([point[0] for point in data], labels, c = labels)
plt.show()

classifier = Perceptron(max_iter = 40)
classifier.fit(data, labels)
print(classifier.score(data, labels))

x_values = [point for point in np.linspace(0,1,100)]
y_values = [point for point in np.linspace(0,1,100)]
point_grid = list(product(x_values, y_values))
distances = classifier.decision_function(point_grid)
distances = abs(distances)

# Heat-map zeichnen
distances_matrix = np.reshape(distances, (100, 100))
heatmap = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(heatmap)
plt.show()
# bei violetter Linie sind die Distancen 0 ==> Decision Boundary
