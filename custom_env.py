from abc import ABC

from rlcard.envs import Env
from rlcard.envs.doudizhu import DoudizhuEnv
from rlcard.games.doudizhu.game import DoudizhuGame


class CustomEnv(DoudizhuEnv, ABC):

    def __init__(self):
        super().__init__({'allow_step_back': True, 'seed': None})

    def run(self, is_training=False):
        """
        This is the exact same function as in the super class, with the only difference being that the env passes
        itself to the agents after each move
        """
        trajectories = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # added line
            self.agents[player_id].set_env(self)

            # Agent plays
            if not is_training:
                action, _ = self.agents[player_id].eval_step(state)
            else:
                action = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
            # Save action
            trajectories[player_id].append(action)

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.num_players):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        return trajectories, payoffs