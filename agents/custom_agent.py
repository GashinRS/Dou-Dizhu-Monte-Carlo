class CustomAgent:

    def __init__(self):
        self.env = None

    def set_env(self, environment):
        self.env = environment

    def step_without_simulation(self, state):
        return min(list(state['legal_actions'].keys()))