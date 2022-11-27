import functools

import numpy as np
from rlcard.games.base import Card
from rlcard.games.doudizhu.player import DoudizhuPlayer
from rlcard.games.doudizhu.judger import DoudizhuJudger
from rlcard.games.doudizhu.utils import cards2str, get_gt_cards, ACTION_2_ID, doudizhu_sort_str

from agents.custom_agent import CustomAgent
from agents.min_agent import MinAgent

JUDGER = DoudizhuJudger([], 0)
INDICES = {0: {1: 0, 2: 1}, 1: {0: 0, 2: 1}, 2: {0: 0, 1: 1}}
OTHER_PLAYERS = {0: [1, 2], 1: [0, 2], 2: [0, 1]}


def make_hand(other_hands_strings):
    other_hands_cards = [[], []]
    for i in range(2):
        # remove the jokers from string of cards because the jokers have a different initialization for Card
        # compared to other cards
        for joker in ["B", "R"]:
            if joker in other_hands_strings[i]:
                # other_hands_strings[i] = other_hands_strings[i].replace(joker, "")
                other_hands_strings[i].remove(joker)
                other_hands_cards[i].append(Card(joker + "J", ''))

        # convert string to Card
        for card in other_hands_strings[i]:
            # the card's suit doesn't matter for determining a hand's decompositions
            other_hands_cards[i].append(Card('S', card))
    return other_hands_cards


def generate_smart_hands_for_opponents(state):
    pid = state['raw_obs']['self']  # player id
    other_players = OTHER_PLAYERS[pid]
    other_card_amount = state['raw_obs']['num_cards_left'].copy()
    other_card_amount.pop(pid)
    remaining_cards = list(state['raw_obs']['others_hand']).copy()

    current_hand = state['raw_obs']['current_hand']
    hasR = "R" in current_hand
    hasB = "B" in current_hand

    other_hands_strings = [[], []]

    for action in state['raw_obs']['trace']:
        if action[0] != pid:
            if not hasR and "R" == action[1] and "B" in remaining_cards:
                other_hands_strings[(INDICES[pid][action[0]] + 1) % 2].append("B")
                remaining_cards.remove("B")
                other_card_amount[(INDICES[pid][action[0]] + 1) % 2] -= 1
            elif not hasB and "B" == action[1] and "R" in remaining_cards:
                other_hands_strings[(INDICES[pid][action[0]] + 1) % 2].append("R")
                remaining_cards.remove("R")
                other_card_amount[(INDICES[pid][action[0]] + 1) % 2] -= 1

    np.random.shuffle(remaining_cards)
    other_hands_strings[0].extend(remaining_cards[:other_card_amount[0]])
    other_hands_strings[1].extend(remaining_cards[other_card_amount[0]:])

    return make_hand(other_hands_strings)


def generate_random_hands_for_opponents(state):
    other_card_amount = state['raw_obs']['num_cards_left'].copy()
    other_card_amount.pop(state['raw_obs']['self'])

    # shuffle the remaining cards of the other players and distribute them
    remaining_cards = list(state['raw_obs']['others_hand'])
    np.random.shuffle(remaining_cards)
    other_hands_strings = [remaining_cards[:other_card_amount[0]],
                           remaining_cards[other_card_amount[0]:]]

    return make_hand(other_hands_strings)


class Node:
    """A node in the MCTS tree"""
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0


class DAgent(CustomAgent):
    def __init__(self, env, rollout_depth):
        super().__init__()
        self.use_raw = False
        self.set_env(env)
        self.rollout_depth = rollout_depth

    def simulate_rollouts(self, state, other_hands_random):
        other_players_index = OTHER_PLAYERS[self.env.game.state['self']]

        other_hands = []
        for i in range(2):
            other_hands.append(self.env.game.players[other_players_index[i]].current_hand)
            self.env.game.players[other_players_index[i]].set_current_hand(other_hands_random[i])

        rollout_depth = self.env.run_for_depth(self.rollout_depth, state, self.env.game.state['self'])
        # TODO calc game score here
        score = 0

        # revert the env back to before the simulation
        for _ in range(rollout_depth):
            self.env.step_back()

        for i in range(2):
            self.env.game.players[other_players_index[i]].set_current_hand(other_hands[i])
        return score

    def calc_opponents_playable_hands_with_randomly_generated_hands(self, state):
        """  Calculates the legal actions opponents can play with randomly generated hands given the current state
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            other_hands_playable_cards (list(int)):
        """
        other_hands_cards = generate_random_hands_for_opponents(state)

        # generate possible plays from the other players
        trace = state['raw_obs']['trace']
        other_hands_playable_cards = [[], []]
        other_players = [DoudizhuPlayer(0, 0), DoudizhuPlayer(1, 0)]
        for i, player in enumerate(other_players):
            player.set_current_hand(other_hands_cards[i])
            if len(trace) > 0:
                offset = 0
                last_played_card = trace[-1][1]
                if last_played_card == "pass":
                    offset += 1
                    last_played_card = trace[-2][1]
                if last_played_card == "pass":  # this means the agent is starting a new round
                    other_hands_playable_cards[i] = JUDGER.playable_cards_from_hand(cards2str(player.current_hand))
                else:
                    last_player_index = INDICES[state['raw_obs']['self']][trace[-1 - offset][0]]
                    greater_player = other_players[last_player_index]
                    greater_player.play(last_played_card)
                    other_hands_playable_cards[i] = get_gt_cards(player, greater_player)
            else:
                other_hands_playable_cards[i] = JUDGER.playable_cards_from_hand(cards2str(player.current_hand))

        # converts raw actions to action ids
        for i in range(2):
            other_hands_playable_cards[i] = list(other_hands_playable_cards[i])
            for j in range(len(other_hands_playable_cards[i])):
                if other_hands_playable_cards[i][j] != 'pass':
                    other_hands_playable_cards[i][j] = ''.join(sorted(other_hands_playable_cards[i][j],
                                                                      key=functools.cmp_to_key(doudizhu_sort_str)))
            other_hands_playable_cards[i] = self.env.get_legal_actions_given_hand(other_hands_playable_cards[i])

        return other_hands_playable_cards

    def step(self, state):
        """  Predict the action given the curent state in gerenerating training data.
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        """
        other_hands_random = generate_random_hands_for_opponents(state)

        # TODO
        """ 
        Selection: select a leaf node from the tree using exploration and exploitation.
        """

        # TODO
        """
        Expansion: expand selected node by all possibles plays.
        """

        """
        Rollout: Simulate rollout out of a state and return score.
        """
        score = self.simulate_rollouts(state)

        # TODO
        """
        Back-Propagation: Update visits and scores of nodes on the path. Also stepback state in each iteration.
        """

        # TODO
        """
        temporay zodat het iets kan doen. Moet dus gefixed worden.
        """
        return MinAgent().step(state)

    def eval_step(self, state):
        """  Predict the action given the current state for evaluation.
            Since this agent is not trained. This function is equivalent to step function
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        """

        return self.step(state), []
