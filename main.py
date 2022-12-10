import functools

import rlcard
from rlcard.agents import RandomAgent
import numpy as np
import torch
from rlcard.games.doudizhu.utils import doudizhu_sort_str

from custom_env import CustomEnv
from agents.custom_random_agent import CustomRandomAgent
from agents.deterministic_agent import DAgent, generate_smart_hands_for_opponents, get_pairs, get_triplets
from agents.min_agent import MinAgent
import matplotlib.pyplot as plt
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)
import time


def main():
    # start = time.time()
    #
    # env = CustomEnv()
    # env.set_agents([DAgent(env, 2),
    #                 MinAgent(), CustomRandomAgent(num_actions=env.num_actions)])
    #
    # scores = np.array([0, 0, 0])
    # for i in range(5):
    #     trajectories, payoffs = env.run()
    #     scores = np.add(scores, payoffs)
    #
    # end = time.time()
    #
    # print(f"score: {scores}, runtime: {end - start}")
    # test_generate_smart_hands_for_opponents()
    # test_DAgent()

    test_generate_smart_hands_for_opponents3()
    # print(get_pairs("3344556"))
    # print(get_triplets("333444556"))


def test_generate_smart_hands_for_opponents():
    state = {'raw_obs': {'num_cards_left': [2, 2, 2], 'self': 0, 'current_hand': '34',
                         'others_hand': 'R678', 'trace': [(1, "B")]}}
    generate_smart_hands_for_opponents(state)


def test_generate_smart_hands_for_opponents2():
    state = {'raw_obs': {'num_cards_left': [4, 4, 5], 'self': 0, 'current_hand': '3456',
                         'others_hand': '899JQKA2B', 'trace': [(2, "777888JQ"), (1, "99")]}}
    generate_smart_hands_for_opponents(state)

def test_generate_smart_hands_for_opponents3():
    state = {'raw_obs': {'num_cards_left': [4, 4, 5], 'self': 1, 'current_hand': '3456',
                         'others_hand': '899JQKAAB', 'trace': [(0, "KKK8"), (2, "AA")]}}
    generate_smart_hands_for_opponents(state)


def test_DAgent():
    env = rlcard.make('doudizhu', {'allow_step_back': True})
    env.set_agents([DAgent(env=env, max_depth=10, num_trees=1, uct_const=1, rollouts=10, default_agent=MinAgent()),
                    RandomAgent(num_actions=env.num_actions), RandomAgent(num_actions=env.num_actions)])

    trajectories, payoffs = env.run()
    print(payoffs)


def test_game(agent_landlord, agent_peasant1, agent_peasant2, its, random_ps=False, random_ll=False):
    env = rlcard.make('doudizhu', {'allow_step_back': True})

    env.set_agents([agent_landlord, agent_peasant1, agent_peasant2])
    if random_ps:
        env.set_agents([agent_landlord, RandomAgent(num_actions=env.num_actions), RandomAgent(num_actions=env.num_actions)])
    if random_ll:
        env.set_agents(
            [RandomAgent(num_actions=env.num_actions), agent_peasant1, agent_peasant2])

    scores = np.array([0, 0, 0])
    for i in range(its):
        trajectories, payoffs = env.run()
        scores = np.add(scores, payoffs)
    print(scores)


def test_agents():
    device = get_device()
    dmc_peasant1 = torch.load("experiments/dmc_result/doudizhu/1_5996800.pth", map_location=device)
    dmc_peasant1.set_device(device)
    dmc_peasant2 = torch.load("experiments/dmc_result/doudizhu/2_5996800.pth", map_location=device)
    dmc_peasant2.set_device(device)
    dmc_landlord = torch.load("experiments/dmc_result/doudizhu/0_5996800.pth", map_location=device)
    dmc_landlord.set_device(device)

    dqn_landlord = torch.load("experiments/doudizhu_dqn/model_landlord_21000.pth")
    # dqn_landlord.set_device(device)

    dqn_peasant1 = torch.load("experiments/doudizhu_dqn/model_peasant1_21000.pth")
    # dqn_peasant1.set_device(device)

    dqn_peasant2 = torch.load("experiments/doudizhu_dqn/model_peasant2_21000.pth")
    # dqn_peasant2.set_device(device)

    test_game(dqn_landlord, MinAgent(), MinAgent(), 100, random_ll=True)


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
