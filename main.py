import constants as c
from innitializer import init_solution
from new_solution import new_solution
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from fitness import fitness

if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    parent = init_solution(c.N)
    child = new_solution(parent, c.N)
    parent_fitness = fitness(parent, x_train, x_test, y_train, y_test)
    child_fitness = fitness(child, x_train, x_test, y_train, y_test)
    print(parent_fitness)
    print(child_fitness)