import numpy as np
from rlcard.games.base import Card
from rlcard.games.doudizhu.player import DoudizhuPlayer
from rlcard.games.doudizhu.judger import DoudizhuJudger
from rlcard.games.doudizhu.utils import cards2str, get_gt_cards

from src.agents.custom_agent import CustomAgent

JUDGER = DoudizhuJudger([], 0)
INDICES = {0: {1: 0, 2: 1}, 1: {0: 0, 2: 1}, 2: {0: 0, 1: 1}}
OTHER_PLAYERS = {0: [1, 2], 1: [0, 2], 2: [0, 1]}


def generate_random_hands_for_opponents(state):
    other_card_amount = state['raw_obs']['num_cards_left'].copy()
    other_card_amount.pop(state['raw_obs']['self'])

    # shuffle the remaining cards of the other players and distribute them
    remaining_cards = list(state['raw_obs']['others_hand'])
    np.random.shuffle(remaining_cards)
    other_hands_strings = [remaining_cards[:other_card_amount[0]],
                           remaining_cards[other_card_amount[0]:]]

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


def calc_opponents_playable_hands_with_randomly_generated_hands(state):
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

    return other_hands_playable_cards


def simulate_rollouts(env, state, rollout_depth):
    handsize = len(env.game.players[0].current_hand)
    other_players_index = OTHER_PLAYERS[env.game.state['self']]

    other_hands = []
    other_hands_random = generate_random_hands_for_opponents(state)
    for i in range(2):
        other_hands.append(env.game.players[other_players_index[i]].current_hand)
        env.game.players[other_players_index[i]].set_current_hand(other_hands_random[i])

    rollout_depth = env.run_for_depth(rollout_depth, state, env.game.state['self'])
    # TODO calc game score here
    score = 0

    # revert the env back to before the simulation
    for _ in range(rollout_depth):
        env.step_back()

    for i in range(2):
        env.game.players[other_players_index[i]].set_current_hand(other_hands[i])
    return score


class DAgent(CustomAgent):

    def __init__(self, env, rollout_depth):
        super().__init__()
        self.use_raw = False
        self.set_env(env)
        self.rollout_depth = rollout_depth

    def step(self, state):
        """  Predict the action given the curent state in gerenerating training data.
        Args:
            state (dict): An dictionary that represents the current state
        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        """
        score = get_env_after_action(self.env, state, self.rollout_depth)

        calc_opponents_playable_hands_with_randomly_generated_hands(state)
        legal_actions = list(state['legal_actions'].keys())
        if len(legal_actions) == 1:
            return legal_actions[0]
        else:
            return min(legal_actions)

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
