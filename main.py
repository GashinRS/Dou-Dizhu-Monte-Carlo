import numpy as np

from custom_env import CustomEnv
from agents.custom_random_agent import CustomRandomAgent
from agents.deterministic_agent import DAgent
from agents.min_agent import MinAgent
import matplotlib.pyplot as plt


def main():

    env = CustomEnv()
    # env = rlcard.make('doudizhu', {'allow_step_back': True})

    # env.set_agents([CustomRandomAgent(num_actions=env.num_actions),
    #                 MinAgent(), CustomRandomAgent(num_actions=env.num_actions)])
    #
    # scores = np.array([0, 0, 0])
    # for i in range(1000):
    #     trajectories, payoffs = env.run()
    #     scores = np.add(scores, payoffs)
    #     # print("==========")
    #     # print(payoffs)
    # print(scores)
    # # set_plt(['Landlord', 'Peasant'], scores[:-1], "MinAgent as landlord\n RandomAgent as peasants", "Games won",
    # #         "1000 games played", "test.png")
    # set_plt(['Landlord', 'Peasant'], scores[:-1], "MinAgent as peasant\n RandomAgent as landlord", "Games won",
    #         "1000 games played", "test.png")

    env.set_agents([DAgent(env, 2),
                    MinAgent(), CustomRandomAgent(num_actions=env.num_actions)])

    scores = np.array([0, 0, 0])
    for i in range(100):
        trajectories, payoffs = env.run()
        scores = np.add(scores, payoffs)
        # print("==========")
        # print(payoffs)
    print(scores)


def set_plt(x, y, xname, yname, plotname, output):
    plt.rcParams["figure.figsize"] = (15, 20)
    plt.rcParams.update({'font.size': 31})
    plt.bar(x, y)
    plt.title(plotname, fontsize=40)
    plt.xlabel(xname, fontsize=40)
    plt.ylabel(yname, fontsize=40)
    plt.savefig(output)
    plt.clf()


if __name__ == "__main__":
    main()
