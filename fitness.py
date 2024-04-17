import numpy as np
from sklearn import svm

def fitness(solution, x_train, x_test, y_train, y_test):
    # Use list comprehensions to create feature subsets based on the solution
    _x_train = [[x_train[i][j] for j in range(len(solution)) if solution[j] == 1] for i in range(len(x_train))]
    _x_test = [[x_test[i][j] for j in range(len(solution)) if solution[j] == 1] for i in range(len(x_test))]

    # Initialize the SVM model with a linear kernel
    model = svm.SVC(kernel="linear")

    # Fit the model on the training data
    model.fit(_x_train, y_train)

    # Evaluate the model on the testing data
    score = model.score(_x_test, y_test)

    return score
