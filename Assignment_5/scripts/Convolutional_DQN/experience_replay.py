from collections import deque
import random


class ReplayMemory:
    def __init__(self, maximum_length):
        self.memory = deque([], maxlen=maximum_length)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
