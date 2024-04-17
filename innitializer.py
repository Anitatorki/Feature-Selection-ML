from random import randint as rnd


def init_solution(n):
    return [rnd(0, 1) for i in range(n)]
