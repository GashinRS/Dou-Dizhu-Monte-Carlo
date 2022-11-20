import rlcard
from rlcard.agents import RandomAgent
import numpy as np

from custom_env import CustomEnv
from agents.custom_random_agent import CustomRandomAgent
from agents.deterministic_agent import DAgent
from agents.min_agent import MinAgent


def main():

    env = CustomEnv()
    # env = rlcard.make('doudizhu', {'allow_step_back': True})

    env.set_agents([DAgent(),
                    MinAgent(), CustomRandomAgent(num_actions=env.num_actions)])

    scores = np.array([0, 0, 0])
    for i in range(10):
        trajectories, payoffs = env.run()
        scores = np.add(scores, payoffs)
        # print("==========")
        # print(payoffs)
    print(scores)


if __name__ == "__main__":
    main()
