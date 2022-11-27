import rlcard
from rlcard.agents import RandomAgent
import numpy as np
import torch
from custom_env import CustomEnv
from agents.custom_random_agent import CustomRandomAgent
from agents.deterministic_agent import DAgent
from agents.min_agent import MinAgent
import matplotlib.pyplot as plt
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)


def main():
    # env = CustomEnv()
    env = rlcard.make('doudizhu', {'allow_step_back': True})

    device = get_device()

    dmc_peasant1 = torch.load("experiments/dmc_result/doudizhu/1_5996800.pth", map_location=device)
    dmc_peasant1.set_device(device)

    dmc_peasant2 = torch.load("experiments/dmc_result/doudizhu/2_5996800.pth", map_location=device)
    dmc_peasant2.set_device(device)

    dmc_landlord = torch.load("experiments/dmc_result/doudizhu/0_5996800.pth", map_location=device)
    dmc_landlord.set_device(device)

    dqn_agent = torch.load("experiments/doudizhu_dqn/model20000.pth")

    #random vs random vs random
    # env.set_agents([CustomRandomAgent(num_actions=env.num_actions), CustomRandomAgent(num_actions=env.num_actions),
    #                 CustomRandomAgent(num_actions=env.num_actions)])

    # random vs random vs random
    # env.set_agents([CustomRandomAgent(num_actions=env.num_actions), MinAgent(), MinAgent()])

    # dmc vs dmc vs dmc
    # env.set_agents([dmc_landlord, dmc_peasant1, dmc_peasant2])

    # dmc vs min_agents
    # env.set_agents([dmc_landlord, MinAgent(), MinAgent()])

    # min_agent vs dmcs
    # env.set_agents([MinAgent(), dmc_peasant1, dmc_peasant2])

    # dmc vs randoms
    # env.set_agents(
    #     [dmc_landlord, CustomRandomAgent(num_actions=env.num_actions), CustomRandomAgent(num_actions=env.num_actions)])

    # random vs dmcs
    # env.set_agents([CustomRandomAgent(num_actions=env.num_actions), dmc_peasant1, dmc_peasant2])

    # dqn vs randoms
    # env.set_agents(
    #     [dqn_agent, CustomRandomAgent(num_actions=env.num_actions), CustomRandomAgent(num_actions=env.num_actions)])

    # dqn vs dmcs
    # env.set_agents([dqn_agent, dmc_peasant1, dmc_peasant2])

    # random vs dqns
    # env.set_agents(
    #     [CustomRandomAgent(num_actions=env.num_actions), dqn_agent, CustomRandomAgent(num_actions=env.num_actions)])

    # dqn vs min_agents
    env.set_agents([dqn_agent, MinAgent(), MinAgent()])

    scores = np.array([0, 0, 0])
    for i in range(1000):
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
