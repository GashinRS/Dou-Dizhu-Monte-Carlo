import rlcard
from rlcard.agents import RandomAgent
from rlcard.models.doudizhu_rule_models import DouDizhuRuleAgentV1
from rlcard.games.doudizhu import judger
from rlcard.games.doudizhu import dealer
from hop_agent import HOPAgent
import numpy as np

from min_agent import MinAgent


def main():
    env = rlcard.make('doudizhu')

    env.set_agents([RandomAgent(num_actions=env.num_actions),
                    MinAgent(), RandomAgent(num_actions=env.num_actions)])

    scores = np.array([0, 0, 0])
    for i in range(500):
        trajectories, payoffs = env.run()
        scores = np.add(scores, payoffs)
        # print("==========")
        # print(payoffs)
    print(scores)


if __name__ == "__main__":
    main()
