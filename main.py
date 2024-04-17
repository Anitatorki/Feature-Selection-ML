import constants as c
from innitializer import init_solution
from new_solution import new_solution
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from fitness import fitness

if __name__ == "__main__":
    # Load the dataset
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize the parent solution and calculate its fitness
    parent = init_solution(c.N)
    parent_fitness = fitness(parent, x_train, x_test, y_train, y_test)

    # Iterate over the specified number of epochs
    for i in range(c.EPOCH):
        # Generate a new child solution and calculate its fitness
        child = new_solution(parent, c.N)
        child_fitness = fitness(child, x_train, x_test, y_train, y_test)

        # Print the current iteration and fitness scores
        print(f"{i + 1} ----> parent: {parent_fitness}, child: {child_fitness}")

        # Update the parent solution if the child has a higher fitness score
        if child_fitness > parent_fitness:
            parent = child.copy()
            parent_fitness = child_fitness

    # Output the best found features and their fitness score
    print(f"Best Found Features: {parent}, Fitness: {parent_fitness}")
