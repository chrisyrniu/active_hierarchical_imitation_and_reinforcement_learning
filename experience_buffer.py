import numpy as np

class ExperienceBuffer():

    def __init__(self, max_buffer_size, batch_size):
        self.size = 0
        self.max_buffer_size = max_buffer_size
        self.experiences = []
        self.batch_size = batch_size

    def add(self, experience):
        assert len(experience) == 7, 'Experience must be of form (s, a, r, s, g, t, grip_info\')'
        assert type(experience[5]) == bool

        self.experiences.append(experience)
        self.size += 1

        # If replay buffer is filled, remove a percentage of replay buffer. 
        if self.size >= self.max_buffer_size:
            beg_index = int(np.floor(self.max_buffer_size/6))
            self.experiences = self.experiences[beg_index:]
            self.size -= beg_index

    def get_batch_with_enc(self):
        states, actions, rewards, new_states, goals, is_terminals, enc = [], [], [], [], [], [], []
        dist = np.random.randint(0, high=self.size, size=min(self.size, self.batch_size))

        for i in dist:
            states.append(self.experiences[i][0])
            actions.append(self.experiences[i][1])
            rewards.append(self.experiences[i][2])
            new_states.append(self.experiences[i][3])
            goals.append(self.experiences[i][4])
            is_terminals.append(self.experiences[i][5])
            enc.append(self.experiences[i][6])

        return states, actions, rewards, new_states, goals, is_terminals, enc

    def get_batch(self):
        states, actions, rewards, new_states, goals, is_terminals = [], [], [], [], [], []
        dist = np.random.randint(0, high=self.size, size=min(self.size, self.batch_size))

        for i in dist:
            states.append(self.experiences[i][0])
            actions.append(self.experiences[i][1])
            rewards.append(self.experiences[i][2])
            new_states.append(self.experiences[i][3])
            goals.append(self.experiences[i][4])
            is_terminals.append(self.experiences[i][5])

        return states, actions, rewards, new_states, goals, is_terminals

    def clear(self, length):
        for i in reversed(range(length)):
            if np.random.random_sample() < 1-np.exp(-i):
                self.experiences.pop()
                self.size -= 1


