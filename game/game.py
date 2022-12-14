import pygame
import rlcard
from rlcard.agents import RandomAgent
import numpy as np
import torch
from agents.deterministic_agent import DAgent, generate_smart_hands_for_opponents
from agents.min_agent import MinAgent
from rlcard.utils import (
    get_device,
)

from player_agent import PlayerAgent


SCREENSIZE = (1300, 900)
SCREENCENTER = (SCREENSIZE[0]/2, SCREENSIZE[1]/2 - 200)
CARDSIZE = (80, 120)


def main():
    # initialising agents and env
    env = rlcard.make('doudizhu', {'allow_step_back': True})
    device = get_device()
    dmc_peasant1 = torch.load("../experiments/dmc_result/doudizhu/dmc_peasant_1.pth", map_location=device)
    dmc_peasant1.set_device(device)
    dmc_peasant2 = torch.load("../experiments/dmc_result/doudizhu/dmc_peasant_2.pth", map_location=device)
    dmc_peasant2.set_device(device)
    dmc_landlord = torch.load("../experiments/dmc_result/doudizhu/dmc_landlord.pth", map_location=device)
    dmc_landlord.set_device(device)

    dqn_landlord = torch.load("../experiments/doudizhu_dqn/model_landlord.pth")
    dqn_peasant1 = torch.load("../experiments/doudizhu_dqn/model_peasant_1.pth")
    dqn_peasant2 = torch.load("../experiments/doudizhu_dqn/model_peasant_2.pth")

    random_agent = RandomAgent(num_actions=env.num_actions)
    min_agent = MinAgent()

    da_agent_landlord = DAgent(env=env, max_depth=3, num_trees=3, uct_const=1, rollouts=100, default_agent=MinAgent(),
                               is_peasant=False)
    da_agent_peasant = DAgent(env=env, max_depth=3, num_trees=3, uct_const=1, rollouts=100, default_agent=MinAgent(),
                              is_peasant=True)

    player_agent = PlayerAgent()

    # setting agents and starting env. player_id has to be the index of the player in the agents array
    player_id = 0
    env.set_agents([player_agent, dqn_peasant1, dqn_peasant2])
    state, pid = env.reset()

    # init pygame and resources
    pygame.init()
    surface = pygame.display.set_mode(SCREENSIZE)
    pygame.display.set_caption("Dou Dizhu")

    icon = pygame.image.load("assets/icon.png")
    pygame.display.set_icon(icon)

    background = pygame.transform.scale(pygame.image.load("assets/background.png"), SCREENSIZE)
    cards = dict()
    cards["R"] = pygame.transform.scale(pygame.image.load("assets/cards/R.png"), CARDSIZE)
    cards["B"] = pygame.transform.scale(pygame.image.load("assets/cards/B.png"), CARDSIZE)
    cards["2"] = pygame.transform.scale(pygame.image.load("assets/cards/2.png"), CARDSIZE)
    cards["A"] = pygame.transform.scale(pygame.image.load("assets/cards/A.png"), CARDSIZE)
    cards["K"] = pygame.transform.scale(pygame.image.load("assets/cards/K.png"), CARDSIZE)
    cards["Q"] = pygame.transform.scale(pygame.image.load("assets/cards/Q.png"), CARDSIZE)
    cards["J"] = pygame.transform.scale(pygame.image.load("assets/cards/J.png"), CARDSIZE)
    cards["T"] = pygame.transform.scale(pygame.image.load("assets/cards/T.png"), CARDSIZE)
    cards["9"] = pygame.transform.scale(pygame.image.load("assets/cards/9.png"), CARDSIZE)
    cards["8"] = pygame.transform.scale(pygame.image.load("assets/cards/8.png"), CARDSIZE)
    cards["7"] = pygame.transform.scale(pygame.image.load("assets/cards/7.png"), CARDSIZE)
    cards["6"] = pygame.transform.scale(pygame.image.load("assets/cards/6.png"), CARDSIZE)
    cards["5"] = pygame.transform.scale(pygame.image.load("assets/cards/5.png"), CARDSIZE)
    cards["4"] = pygame.transform.scale(pygame.image.load("assets/cards/4.png"), CARDSIZE)
    cards["3"] = pygame.transform.scale(pygame.image.load("assets/cards/3.png"), CARDSIZE)
    
    back = pygame.transform.scale(pygame.image.load("assets/cards/back.png"), CARDSIZE)
    landlord = pygame.transform.scale(pygame.image.load("assets/landlord.png"), (70, 70))
    peasant = pygame.transform.scale(pygame.image.load("assets/peasant.png"), (70, 70))

    large_font = pygame.font.Font('freesansbold.ttf', 64)
    smoll_font = pygame.font.Font('freesansbold.ttf', 24)

    defeat_text = large_font.render('DEFEAT', True, (0, 0, 0))
    defeat_text_rect = defeat_text.get_rect()
    defeat_text_rect.center = SCREENCENTER
    victory_text = large_font.render('VICTORY', True, (0, 0, 0))
    victory_text_rect = victory_text.get_rect()
    victory_text_rect.center = SCREENCENTER

    seen_cards = env.game.state["seen_cards"]

    # pygame loop
    running = True
    while running:

        surface.blit(background, (0, 0))

        if player_id == 0:
            surface.blit(peasant, (10, 10))
            surface.blit(peasant, (SCREENSIZE[0] - 80, 10))
        elif player_id == 1:
            surface.blit(landlord, (10, 10))
            surface.blit(peasant, (SCREENSIZE[0] - 80, 10))
        else:
            surface.blit(peasant, (10, 10))
            surface.blit(landlord, (SCREENSIZE[0] - 80, 10))
        surface.blit(large_font.render(str(env.game.state['num_cards_left'][player_id-1]), True, (0, 0, 0)), (10, 90))
        surface.blit(large_font.render(str(env.game.state['num_cards_left'][player_id - 2]), True, (0, 0, 0)), (SCREENSIZE[0] - 80, 90))

        if pid == (player_id + 1) % 3:
            pygame.draw.polygon(surface, color=(239, 245, 66), points=[(SCREENSIZE[0] - 150, 10), (SCREENSIZE[0] - 150, 20), (SCREENSIZE[0] - 140, 15)])
        elif pid == (player_id + 2) % 3:
            pygame.draw.polygon(surface, color=(239, 245, 66), points=[(150, 10), (150, 20), (140, 15)])
        else:
            pygame.draw.polygon(surface, color=(239, 245, 66), points=[(SCREENCENTER[0] - 5, SCREENSIZE[1] - 200),
                                                                       (SCREENCENTER[0] + 5, SCREENSIZE[1] - 200),
                                                                       (SCREENCENTER[0], SCREENSIZE[1] - 190)])

        last_plays_by_player = ["", "", ""]
        last_3_plays = []
        j = 1
        while j < 4 and j <= len(env.game.state['trace']):
            play = env.game.state['trace'][-j]
            last_3_plays.append(play)
            last_plays_by_player[play[0]] = play[1]
            j += 1

        surface.blit(smoll_font.render(last_plays_by_player[player_id-1], True, (0, 0, 0)), (10, 150))

        surface.blit(smoll_font.render(last_plays_by_player[player_id-2], True, (0, 0, 0)), (SCREENSIZE[0] - 150, 150))

        s = (SCREENSIZE[0] - (CARDSIZE[0] - 20) * 3) / 2
        for i, c in enumerate(seen_cards):
            surface.blit(cards[c], (s + (CARDSIZE[0] - 20) * i, 30))

        s = (SCREENSIZE[0] - (CARDSIZE[0] - 20)*len(env.game.players[player_id].current_hand))/2
        for i, card in enumerate(env.game.players[player_id].current_hand):
            c = card.rank
            if c == '':
                c = card.suit[0]
            surface.blit(cards[c], (s + (CARDSIZE[0] - 20)*i, SCREENSIZE[1] - CARDSIZE[1] - 30))

        last_play = ""
        j = 0
        while (last_play == "" or last_play == "pass") and j < len(last_3_plays):
            if j == 2 and last_play == "pass":
                last_play = ""
                break
            last_play = last_3_plays[j][1]
            j += 1

        s = (SCREENSIZE[0] - (CARDSIZE[0] - 20) * len(last_play)) / 2
        for i, c in enumerate(last_play):
            surface.blit(cards[c], (s + (CARDSIZE[0] - 20) * i, SCREENSIZE[1]/2))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if env.game.is_over():
            if player_id == env.game.winner_id:
                surface.blit(victory_text, victory_text_rect)
            else:
                surface.blit(defeat_text, defeat_text_rect)
        else:
            pygame.display.update()
            if pid == player_id:
                state, pid = env.step(env.agents[pid].step(state), True)
            else:
                state, pid = env.step(env.agents[pid].step(state))

        pygame.display.update()


if __name__ == '__main__':
    main()
