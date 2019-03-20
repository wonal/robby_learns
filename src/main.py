import src.environment as env
from src.agent import Problem


if __name__ == '__main__':
    p = Problem(env.GRID_BOUND, env.GRID_BOUND)
    mean, std = p.run(5000, 200)
    print("mean: {}, standard deviation: {}".format(mean, std))

