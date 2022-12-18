import rlcard
from rlcard.agents import RandomAgent
import numpy as np
import torch
from agents.deterministic_agent import DAgent, generate_smart_hands_for_opponents
from agents.min_agent import MinAgent
import matplotlib.pyplot as plt
from multiprocessing import Process
from rlcard.utils import (
    get_device,
)


def main():
    test_agents()
    # set_plt(["landlord", "peasant"], [398, 602], "DQN as landlord\nRandomAgent as peasants", "Games Won", "1000 Games played", "plots/dqn_perf.png")
    # set_plt(["landlord", "peasant"], [89, 194], "DA as landlord\nDMC as peasants", "Games Won",
    #         f"{89 + 194} Games played", "plots/davsdmc.png")
    #
    # set_plt(["landlord", "peasant"], [189, 38], "DMC as landlord\nDA as peasants", "Games Won",
    #         f"{189 + 38} Games played", "plots/dmcvsda.png")
    #
    # set_plt(["landlord", "peasant"], [209, 95], "DA as landlord\nDQN as peasants", "Games Won",
    #         f"{209 + 95} Games played", "plots/davsdqn.png")
    #
    # set_plt(["landlord", "peasant"], [34, 131], "DQN as landlord\nDA as peasants", "Games Won",
    #         f"{34 + 131} Games played", "plots/dqnvsda.png")
    #
    #
    # set_plt(["landlord", "peasant"], [312, 88], "DA as landlord\nRandom Agent as peasants", "Games Won",
    #         f"{312 + 88} Games played", "plots/davsrand.png")
    #
    # set_plt(["landlord", "peasant"], [34, 173], "Random Agent as landlord\nDA as peasants", "Games Won",
    #         f"{34 + 173} Games played", "plots/randvsda.png")
    #
    # set_plt(["landlord", "peasant"], [236, 164], "DA as landlord\nMinAgent as peasants", "Games Won",
    #         f"{236 + 164} Games played", "plots/davsmin.png")
    #
    # set_plt(["landlord", "peasant"], [93, 87], "MinAgent as landlord\nDA as peasants", "Games Won",
    #         f"{93 + 87} Games played", "plots/minvsda.png")

    # set_plt(["random", "dqn", "MinAgent", "dmc"], [.78, .69, .59, .31],
    #         "Winrates of Deterministic MTSC agent as landlord", "Win rate", f"", "plots/vsalllandlord.png")
    #
    # set_plt(["random", "dqn", "MinAgent", "dmc"], [.83, .79, .48, .17],
    #         "Winrates of Deterministic MTSC agent as peasants", "Win rate", f"", "plots/vsallaspeasants.png")




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
    device = get_device()
    env = rlcard.make('doudizhu', {'allow_step_back': True, "allow_raw_data": True})
    # env = CustomEnv()
    env.set_agents([DAgent(env=env, max_depth=3, num_trees=1, uct_const=1, rollouts=200, default_agent=MinAgent()),
                    RandomAgent(num_actions=env.num_actions), RandomAgent(num_actions=env.num_actions)
                    # MinAgent(), MinAgent()
                    ])
    scores = np.array([0, 0, 0])
    for i in range(100):
        trajectories, payoffs = env.run()
        print(f"{i}: {[payoffs]}")
        scores = np.add(scores, payoffs)
    print(scores)


def test_game(env, agent_landlord, agent_peasant1, agent_peasant2, its):
    env.set_agents([agent_landlord, agent_peasant1, agent_peasant2])

    scores = np.array([0, 0, 0])
    for i in range(its):
        trajectories, payoffs = env.run()
        scores = np.add(scores, payoffs)
        # print(f"{i}: {[scores]}")
    print(scores)


def test_agents():
    env = rlcard.make('doudizhu', {'allow_step_back': True})
    device = get_device()
    dmc_peasant1 = torch.load("experiments/dmc_result/doudizhu/dmc_peasant_1.pth", map_location=device)
    dmc_peasant1.set_device(device)
    dmc_peasant2 = torch.load("experiments/dmc_result/doudizhu/dmc_peasant_2.pth", map_location=device)
    dmc_peasant2.set_device(device)
    dmc_landlord = torch.load("experiments/dmc_result/doudizhu/dmc_landlord.pth", map_location=device)
    dmc_landlord.set_device(device)

    dmc_pretrained_landlord = torch.load("experiments/dmc_result/dmc_pretrained/0.pth", map_location=device)
    dmc_pretrained_landlord.set_device(device)
    dmc_pretrained_peasant1 = torch.load("experiments/dmc_result/dmc_pretrained/1.pth", map_location=device)
    dmc_pretrained_peasant1.set_device(device)
    dmc_pretrained_peasant2 = torch.load("experiments/dmc_result/dmc_pretrained/2.pth", map_location=device)
    dmc_pretrained_peasant2.set_device(device)

    dqn_landlord = torch.load("experiments/doudizhu_dqn/model_landlord.pth")
    # dqn_landlord.set_device(device)

    dqn_peasant1 = torch.load("experiments/doudizhu_dqn/model_peasant_1.pth")
    # dqn_peasant1.set_device(device)

    dqn_peasant2 = torch.load("experiments/doudizhu_dqn/model_peasant_2.pth")
    # dqn_peasant2.set_device(device)

    random_agent = RandomAgent(num_actions=env.num_actions)

    mc_agent_landlord = DAgent(env=env, max_depth=3, num_trees=3, uct_const=1, rollouts=200, default_agent=MinAgent(),
                               is_peasant=False)
    mc_agent_peasant = DAgent(env=env, max_depth=3, num_trees=3, uct_const=1, rollouts=200, default_agent=MinAgent(),
                              is_peasant=True)

    print("dqn vs rand")
    test_game(env, dqn_landlord, random_agent, random_agent, 1000)
    print("rand vs dqn")
    test_game(env, random_agent, dqn_peasant1, dqn_peasant2, 1000)
    print("min vs rand")
    test_game(env, MinAgent(), random_agent, random_agent, 1000)
    print("rand vs min")
    test_game(env, random_agent, MinAgent(), MinAgent(), 1000)
    print("dmc vs rand")
    test_game(env, dmc_landlord, random_agent, random_agent, 1000)
    print("rand vs dmc")
    test_game(env, random_agent, dmc_peasant1, dmc_peasant2, 1000)
    # test_game(env, dmc_pretrained_landlord, dmc_peasant1, dmc_peasant2, 100)
    # test_game(env, dmc_landlord, dmc_pretrained_peasant1, dmc_pretrained_peasant1, 100)
    # test_game(env, mc_agent_landlord, MinAgent(), MinAgent(), 200)
    # test_game(env, dmc_landlord, random_agent, random_agent, 1000)
    # test_game(env, dqn_landlord, random_agent, random_agent, 1000)


def set_plt(x, y, xname, yname, plotname, output):
    plt.rcParams["figure.figsize"] = (40, 20)
    plt.rcParams.update({'font.size': 31})
    plt.bar(x, y, width=0.4)
    plt.title(plotname, fontsize=40)
    plt.xlabel(xname, fontsize=40)
    plt.ylabel(yname, fontsize=40)
    plt.savefig(output)
    plt.clf()


def plots():
    plt.show()
    # %%
    landlord = [.78, .69, .59, .31]
    peasant = [.83, .79, .48, .17]

    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(landlord))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot

    plt.bar(br1, landlord, color='r', width=barWidth,
            edgecolor='grey', label='Landlord')
    plt.bar(br2, peasant, color='b', width=barWidth,
            edgecolor='grey', label='Peasant')

    # Adding Xticks
    plt.title("MCA win rates als landlord en peasant", fontweight='bold', fontsize=15)
    plt.xlabel('opponent(s)', fontweight='bold', fontsize=15)
    plt.ylabel('Win Rate', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth / 2 for r in range(len(landlord))],
               ['random', 'DQN', 'minimum', 'DMC'])

    plt.legend()
    plt.savefig("grafieken/potatoaganet.png")

    plt.savefig("grafieken/potatoaganet.png")
    # %%
    landlord = [.382, .779, .925]
    peasant = [.636, .893, .946]

    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(landlord))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot

    plt.bar(br1, landlord, color='r', width=barWidth,
            edgecolor='grey', label='Landlord')
    plt.bar(br2, peasant, color='b', width=barWidth,
            edgecolor='grey', label='Peasant')

    # Adding Xticks
    plt.title("Win rates van verschillende agents tegen random", fontweight='bold', fontsize=15)
    plt.xlabel('opponent(s)', fontweight='bold', fontsize=15)
    plt.ylabel('Win Rate', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth / 2 for r in range(len(landlord))],
               ['DQN', 'minimum', 'DQN'])

    plt.legend()
    plt.savefig("grafieken/agents.png")


if __name__ == "__main__":
    main()
    # plots()