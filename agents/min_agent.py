"""
This agent always plays the smallest possible cards to ensure big combinations aren't wasted
"""
from src.agents.custom_agent import CustomAgent


class MinAgent(CustomAgent):

    def __init__(self):
        super().__init__()
        self.use_raw = False

    @staticmethod
    def step(state):
        return min(list(state['legal_actions'].keys()))

    def eval_step(self, state):
        return self.step(state), []
