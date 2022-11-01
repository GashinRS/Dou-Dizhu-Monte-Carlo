"""
This agent always plays the smallest possible cards to ensure big combinations aren't wasted
"""


class MinAgent(object):

    def __init__(self):
        self.use_raw = False

    @staticmethod
    def step(state):
        return min(list(state['legal_actions'].keys()))

    def eval_step(self, state):
        return self.step(state), []
