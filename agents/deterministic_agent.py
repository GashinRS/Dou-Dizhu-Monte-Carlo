import functools
import random

import numpy as np
from rlcard.games.base import Card
from rlcard.games.doudizhu.player import DoudizhuPlayer
from rlcard.games.doudizhu.judger import DoudizhuJudger
from rlcard.games.doudizhu.utils import cards2str, get_gt_cards, ACTION_2_ID, doudizhu_sort_str, CARD_RANK_STR_INDEX, \
    CARD_RANK_STR

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
    other_card_amount = state['raw_obs']['num_cards_left'].copy()
    other_card_amount.pop(pid)
    remaining_cards = list(state['raw_obs']['others_hand']).copy()

    current_hand = state['raw_obs']['current_hand']
    hasR = "R" in current_hand
    hasB = "B" in current_hand

    other_hands_strings = [[], []]

    for action in state['raw_obs']['trace']:
        if action[0] != pid and action[1] != "pass":
            if not hasR and "R" == action[1] and "B" in remaining_cards:
                other_hands_strings[(INDICES[pid][action[0]] + 1) % 2].append("B")
                remaining_cards.remove("B")
                other_card_amount[(INDICES[pid][action[0]] + 1) % 2] -= 1
            elif not hasB and "B" == action[1] and "R" in remaining_cards:
                other_hands_strings[(INDICES[pid][action[0]] + 1) % 2].append("R")
                remaining_cards.remove("R")
                other_card_amount[(INDICES[pid][action[0]] + 1) % 2] -= 1

            paircards = get_pairs(action[1])
            tripletcards = get_triplets(action[1])
            paircards_copy = paircards.copy()

            # when someone plays higher pairs (we chose for pairs of K, A and 2) it not highly unlikely that they
            # didn't have a triplet
            KA2 = []
            if "K" in paircards_copy:
                KA2.append("K")
            if "A" in paircards_copy:
                KA2.append("A")
            if "2" in paircards_copy:
                KA2.append("2")

            for card in KA2:
                paircards_copy.remove(card)
                if remaining_cards.count(card) == 2:
                    other_hands_strings[(INDICES[pid][action[0]] + 1) % 2].append(card)
                    remaining_cards.remove(card)
                    other_card_amount[(INDICES[pid][action[0]] + 1) % 2] -= 1

            for samecards in [paircards_copy, tripletcards]:
                if len(samecards) > 0:
                    # contains the counts of how many cards of the pair/triplet are left in the remaining cards
                    remaining_cards_count = [0] * len(samecards)

                    for i, card in enumerate(samecards):
                        remaining_cards_count[i] = remaining_cards.count(card)

                    for i, count in enumerate(remaining_cards_count):
                        if count > 0:
                            for j in range(count):
                                other_hands_strings[(INDICES[pid][action[0]] + 1) % 2].append(samecards[i])
                                remaining_cards.remove(samecards[i])
                                other_card_amount[(INDICES[pid][action[0]] + 1) % 2] -= 1

            # if someone played a triplet paired wit a single card we'll assume he/she has no more cards that are the
            # same as the single
            singles = []
            if len(tripletcards) > 0 and len(paircards) == 0:
                for card in tripletcards:
                    hand = [*action[1]]
                    for _ in range(3):
                        hand.remove(card)
                    for card_in_hand in hand:
                        singles.append(card_in_hand)

                for card in singles:
                    while card in remaining_cards:
                        other_hands_strings[(INDICES[pid][action[0]] + 1) % 2].append(card)
                        remaining_cards.remove(card)
                        other_card_amount[(INDICES[pid][action[0]] + 1) % 2] -= 1

    np.random.shuffle(remaining_cards)
    other_hands_strings[0].extend(remaining_cards[:other_card_amount[0]])
    other_hands_strings[1].extend(remaining_cards[other_card_amount[0]:])
    # print(other_hands_strings)
    return make_hand(other_hands_strings)


def extract_same_cards(hand, amount):
    cardcount = [0] * 15
    for card in hand:
        cardcount[CARD_RANK_STR_INDEX[card]] += 1

    samecards = []  # contains the card of which there was a pair in the given hand

    for card in CARD_RANK_STR:
        if cardcount[CARD_RANK_STR_INDEX[card]] == amount:
            samecards.append(card)

    return samecards


def get_pairs(hand):
    return extract_same_cards(hand, 2)


def get_triplets(hand):
    return extract_same_cards(hand, 3)


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
    def __init__(self, env, max_depth, num_trees, uct_const, rollouts, default_agent, is_peasant):
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
        self.roots = []
        self.is_peasant = is_peasant

    def simulate_rollouts(self, state, node):
        i = 0
        while i < self.max_depth and not self.env.game.is_over():
            if len(self.env.game.state["actions"]) == 0:
                self.env.game.state["actions"] = list(self.env.game.state["current_hand"])
                state["raw_legal_actions"] = list(self.env.game.state["current_hand"])
                state, pid = self.env.step(self.default_agent.step_raw(state), True)
            else:
                state, pid = self.env.step(self.default_agent.step(state))
            i += 1

        score = self.calculate_score(node)
        for _ in range(i):
            self.env.step_back()
        return score

    def calculate_score_naive(self):
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

    def calculate_score(self, node):
        if self.env.game.is_over():
            if self.pid == self.env.game.winner_id:
                return 1000000
            else:
                return -1000000
        else:
            score = 0
            player = self.env.game.players[self.pid]
            for card in self.env.game.state["current_hand"]:
                score += CARD_RANK_STR_INDEX[card] + 1

            if node.action == "pass":
                # return score / np.power(len(player.current_hand), 4)
                return 0
            else:
                return score / len(player.current_hand)

    def calculate_score_simple(self, node):
        if self.env.game.is_over():
            if self.pid == self.env.game.winner_id:
                return 100
            else:
                return -100
        return 0


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
        og_state = state

        self.pid = self.env.get_player_id()
        self.roots = []
        for _ in range(self.num_trees):
            state = og_state
            other_hands = generate_smart_hands_for_opponents(state)

            other_players_index = OTHER_PLAYERS[self.env.game.state['self']]

            other_hands_real = []
            for i in range(2):
                other_hands_real.append(self.env.game.players[other_players_index[i]].current_hand)
                self.env.game.players[other_players_index[i]].set_current_hand(other_hands[i])

            root = Node(None, None)
            pnode = Node(None, None)
            for _ in range(self.rollouts):
                """ 
                Selection: select a leaf node from the tree using exploration and exploitation.
                """
                node = self.select_node(root)
                score = self.calculate_score(pnode)

                if not self.env.game.is_over():
                    """
                    Expansion: expand selected node by all possibles plays.
                    """
                    actions = self.env.game.state['actions']
                    if len(actions) == 0:
                        actions = list(self.env.game.state["current_hand"])
                    node.children = [Node(node, a) for a in actions]

                    """
                    Rollout: Simulate rollout out of a state and return score.
                    """
                    node = random.choice(node.children)
                    state, pid = self.env.step(node.action, True)

                    if self.env.game.state["self"] == (self.pid + 1) % 3:
                        pnode = node

                    score = self.simulate_rollouts(state, pnode)

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

            self.roots.append(root)
            # rollouts for loop
        # Trees for loop

        res_root = self.roots[0]
        for i in range(1, len(self.roots)):
            for j in range(len(self.roots[i].children)):
                res_root.children[j].value += self.roots[i].children[j].value
                res_root.children[j].visits += self.roots[i].children[j].visits

        to_play_node = res_root.children[0]
        for child in res_root.children:
            if self.uct_last_choice(child) > self.uct_last_choice(to_play_node):
                to_play_node = child

        return self.ACTION_2_ID[to_play_node.action]
        # return to_play_node.action

    def uct(self, node):
        """Upper Confidence Bound for trees formule"""
        if node.visits > 0 and node.parent is not None:
            return (node.value + 2 * self.uct_const * np.sqrt(2 * np.log(node.parent.visits) / np.power(node.visits, 4))) * node.game_over
        else:
            return np.inf

    def uct_last_choice(self, node):
        """Upper Confidence Bound for trees formule"""
        if node.visits > 0 and node.parent is not None:
            return node.value / node.visits
        else:
            return 0

    def select_node(self, root):
        """
        selects a node using the exploit and explore strategy
        :param root: root of mcts tree
        :return: leaf node
        """
        node = root
        i = 0
        while len(node.children) > 0:
            newnode = node.children[0]
            for child in node.children:
                if not self.is_peasant:
                    if i % 3 == 0:
                        if self.uct(child) > self.uct(newnode):
                            newnode = child
                    else:
                        if self.uct(child) < self.uct(newnode):
                            newnode = child
                else:
                    if i % 3 == 0:
                        if self.uct(child) < self.uct(newnode):
                            newnode = child
                    else:
                        if self.uct(child) > self.uct(newnode):
                            newnode = child
            node = newnode
            self.env.step(node.action, True)
            i += 1
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
