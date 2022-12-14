from agents.custom_agent import CustomAgent


class PlayerAgent(CustomAgent):

    def __init__(self):
        super().__init__()

    @staticmethod
    def step(state):
        move = input("input move: ")
        while move not in state['raw_legal_actions']:
            print("illegal move")
            move = input("input move: ")

        return move

    def eval_step(self, state):
        return self.step(state), []
