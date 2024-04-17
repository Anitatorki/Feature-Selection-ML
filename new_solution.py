from random import randint as rnd


def new_solution(parent, n):
    child = parent.copy()
    cell = rnd(0, n-1)
    child[cell] = 1 if child[cell]==0 else 0
    return child


