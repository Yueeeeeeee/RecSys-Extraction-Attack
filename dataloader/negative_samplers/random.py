from .base import AbstractNegativeSampler
from tqdm import trange
import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        num_samples = 2 * self.user_count * self.sample_size
        all_samples = np.random.choice(self.item_count, num_samples) + 1

        seen_samples = {}
        negative_samples = {}
        print('Sampling negative items randomly...')
        j = 0
        for i in trange(self.user_count):
            user = i + 1
            seen = set(self.train[user])
            seen.update(self.val[user])
            seen.update(self.test[user])
            seen_samples[user] = seen

            samples = []
            while len(samples) < self.sample_size:
                item = all_samples[j % num_samples]
                j += 1
                if item in seen or item in samples:
                    continue
                samples.append(item)
            negative_samples[user] = samples

        return seen_samples, negative_samples
