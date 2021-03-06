#wrappy-svm
A Python class which provides Linear Support Vector Machine classification by
interfacing with Octave.

##Requirements
Runs on Python 2, since at least version 2.7.6. Requires GNU Octave, tested with
version 3.8.1.

##Usage

```python
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
```
