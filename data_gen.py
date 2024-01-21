import numpy as np


# Function to generate a synthetic dataset for classification
def generate_classification_data(num_samples, num_features, num_classes):
    data = np.random.randn(num_samples, num_features)
    labels = np.random.randint(0, num_classes, num_samples)
    return data, labels

num_samples = 1000
num_features = 3
num_classes = 5
data, labels = generate_classification_data(num_samples, num_features, num_classes)

with open('dataset.txt', 'w') as file:
    for pt in data:
        items = [str(i) for i in pt]
        file.write(str(','.join(items)) + '\n')

with open('labels.txt', 'w') as file:
    for l in labels:
        file.write(str(l) + '\n')


