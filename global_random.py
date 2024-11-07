import random

class GlobalRandom:
    def __init__(self, seed=42):
        self.current_seed = seed
        self.random = random.Random(seed)

    def seed(self, seed):
        self.current_seed = seed
        self.random.seed(seed)

    def get_current_seed(self):
        return self.current_seed

    def randint(self, a, b):
        return self.random.randint(a, b)

    def sample(self, population, k):
        return self.random.sample(population, k)

    def choice(self, seq):
        return self.random.choice(seq)

    def shuffle(self, x):
        return self.random.shuffle(x)

global_random = GlobalRandom()