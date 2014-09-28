#!/usr/bin/env python

import wrappy_svm

# x - instance attribute tuples
x = [(0.3, 0.4), (0.4, 0.6), (0.9, 0.4), (0.7, 0.8)]

# y - instance classes (1 or -1)
y = [1, -1, -1, -1]

# train the classifier
classifier = wrappy_svm.svm(x, y)
classifier.train()

# get the equation of the hyperplane
print(classifier.hyperplane())

# classify new tuples
print(classifier.classify([0.5, 0.5]))
