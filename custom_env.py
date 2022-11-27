from abc import ABC

from rlcard.envs.doudizhu import DoudizhuEnv, _cards2array

class CustomEnv(DoudizhuEnv, ABC):

    def __init__(self):
        super().__init__({'allow_step_back': True, 'seed': 42})

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

            trajectories, player_id, state = self.set_next_state(trajectories, player_id, action)

        # Add a final state to all the players
        for player_id in range(self.num_players):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        return trajectories, payoffs

    def run_for_depth(self, rollout_depth, state, player_id):
        """  Runs the env given an action
        Args:
            :param rollout_depth: the amount of turns the env is going to run
            :param state: the current state
            :param player_id: the player's id
        Returns:
            rollout_depth (int)
        """
        trajectories = [[] for _ in range(self.num_players)]

        # Loop to play the game
        trajectories[player_id].append(state)
        depth = 0
        while not self.is_over() and depth < rollout_depth:
            depth += 1

            # Agent plays
            action = self.agents[player_id].step_without_simulation(state)

            trajectories, player_id, state = self.set_next_state(trajectories, player_id, action)

        # Add a final state to all the players
        for player_id in range(self.num_players):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        return depth

    def run_given_action(self, state, player_id, action):
        """  Runs the env given an action
        Args:
            :param action: the id of the action
            :param state: the current state
            :param player_id: the player's id
        """
        trajectories = [[] for _ in range(self.num_players)]

        trajectories[player_id].append(state)

        trajectories, player_id, state = self.set_next_state(trajectories, player_id, action)

        # Add a final state to all the players
        for player_id in range(self.num_players):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

    def set_next_state(self, trajectories, player_id, action):
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

        return trajectories, player_id, state

    # def get_legal_actions_given_hand(self, legal_actions):
    #     """ Get all legal actions given the raw actions
    #     Args:
    #         legal_actions (list(string)): raw legal actions
    #     Returns:
    #         legal_actions (dict): a list of legal actions' id
    #     """
    #     return {self._ACTION_2_ID[action]: _cards2array(action) for action in legal_actions}

    def get_legal_actions_given_hand(self, legal_actions):
        """ Get all legal actions given the raw actions
        Args:
            legal_actions (list(string)): raw legal actions
        Returns:
            legal_actions (list(int)): a list of legal actions' id
        """
        return [self._ACTION_2_ID[action] for action in legal_actions]
