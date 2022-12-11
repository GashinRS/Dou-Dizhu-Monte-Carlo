"""
This agent always plays the smallest possible cards to ensure big combinations aren't wasted
"""
from agents.custom_agent import CustomAgent


class MinAgent(CustomAgent):

    def __init__(self):
        super().__init__()
        self.use_raw = False

    @staticmethod
    def step(state):
        # test = state['legal_actions']
        # testkeys = list(state['legal_actions'].keys())
        # testraw = state['raw_legal_actions'][testkeys.index(min(testkeys))]
        return min(list(state['legal_actions'].keys()))
        # return testraw

    @staticmethod
    def step_raw(state):
        return state['raw_legal_actions'][0]

    def eval_step(self, state):
        return self.step(state), []
