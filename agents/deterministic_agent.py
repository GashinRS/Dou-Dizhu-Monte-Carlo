import functools
import random

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
        self.game_over = 1 # 1 if game is not over when doing this action


class DAgent(CustomAgent):
    def __init__(self, env, max_depth, num_trees, uct_const, rollouts, default_agent):
        from rlcard.games.doudizhu.utils import ACTION_2_ID
        self.ACTION_2_ID = ACTION_2_ID
        super().__init__()
        self.pid = None
        self.use_raw = False
        self.set_env(env)
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.uct_const = uct_const
        self.rollouts = rollouts
        self.default_agent = default_agent
        self.debug = 0

    def simulate_rollouts(self, state):
        # other_players_index = OTHER_PLAYERS[self.env.game.state['self']]
        #
        # other_hands = []
        # for i in range(2):
        #     other_hands.append(self.env.game.players[other_players_index[i]].current_hand)

        # rollout_depth = self.env.run_for_depth(self.max_depth, state, self.env.game.state['self'])

        i = 0
        self.debug += 1
        print(self.debug)
        while i < self.max_depth and not self.env.game.is_over():
            state, pid = self.env.step(self.default_agent.step(state))
            i += 1

        # TODO calc game score here
        score = self.calculate_score()

        # revert the env back to before the simulation
        # for _ in range(rollout_depth):
        #     self.env.step_back()

        for _ in range(i):
            self.env.step_back()

        # for i in range(2):
        #     self.env.game.players[other_players_index[i]].set_current_hand(other_hands[i])
        return score

    def calculate_score(self):
        if self.env.game.is_over():
            if self.pid == self.env.game.winner_id:
                return 100
            else:
                return -100
        else:
            player = self.env.game.players[self.pid]
            player_up = self.env.game.players[(player.player_id + 1) % len(self.env.game.players)]
            player_down = self.env.game.players[(player.player_id - 1) % len(self.env.game.players)]
            return len(player_up.current_hand) + len(player_down.current_hand) - (len(player.current_hand) * 2)


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

        self.pid = self.env.get_player_id()

        other_hands_random = generate_random_hands_for_opponents(state)

        other_players_index = OTHER_PLAYERS[self.env.game.state['self']]

        other_hands_real = []
        for i in range(2):
            other_hands_real.append(self.env.game.players[other_players_index[i]].current_hand)
            self.env.game.players[other_players_index[i]].set_current_hand(other_hands_random[i])

        root = Node(None, None)
        potatosaladondfireisverysadbecauseiwantedtoeatthepotatosaladandnowicantsadge = 0
        for _ in range(self.rollouts):
            # print(potatosaladondfireisverysadbecauseiwantedtoeatthepotatosaladandnowicantsadge)
            potatosaladondfireisverysadbecauseiwantedtoeatthepotatosaladandnowicantsadge += 1
            """ 
            Selection: select a leaf node from the tree using exploration and exploitation.
            """
            node = self.select_node(root)

            """
            Expansion: expand selected node by all possibles plays.
            """
            actions = self.env.game.state['actions']
            for _ in range(self.max_depth):
                # TODO maybe use numpy for optimisation
                node.children.append(Node(node, random.choice(actions)))

            """
            Rollout: Simulate rollout out of a state and return score.
            """
            node = random.choice(node.children)
            state, pid = self.env.step(node.action, True)

            while self.env.game.is_over():
                node.game_over = 0
                node = random.choice(node.children)
                state, pid = self.env.step(node.action, True)

            score = self.simulate_rollouts(state)

            """
            Back-Propagation: Update visits and scores of nodes on the path. Also stepback state in each iteration.
            """
            while node.parent is not None:
                node.visits += 1
                node.value += score
                node = node.parent
                self.env.step_back()
            node.value += score
            node.visits += 1

        """
        Zet kaarten in de env terug op originele
        """
        for i in range(2):
            self.env.game.players[other_players_index[i]].set_current_hand(other_hands_real[i])

        to_play_node = root.children[0]
        for child in root.children:
            if child.value > to_play_node.value:
                to_play_node = child

        return self.ACTION_2_ID[to_play_node.action]
        # return to_play_node.action

    def uct(self, node):
        """Upper Confidence Bound for trees formule"""
        if node.visits > 0 and node.parent is not None:
            return (node.value + 2 * self.uct_const * np.sqrt(2 * np.log(node.parent.visits) / node.visits)) * node.game_over
        else:
            return np.inf

    def select_node(self, root):
        """
        selects a node using the exploit and explore strategy
        :param root: root of mcts tree
        :return: leaf node
        """
        node = root
        while len(node.children) > 0:
            newnode = node.children[0]
            for child in node.children:
                if self.uct(child) > self.uct(newnode):
                    newnode = child
            node = newnode
            self.env.step(node.action, True)
            # state = self.env.run_given_action(state, node.action)
        return node

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
