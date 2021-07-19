from .base import AbstractNegativeSampler
from tqdm import trange
from collections import Counter
import numpy as np


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        popularity = self.items_by_popularity()
        items = list(popularity.keys())
        total = 0
        for i in range(len(items)):
            total += popularity[items[i]]
        for i in range(len(items)):
            popularity[items[i]] /= total
        probs = list(popularity.values())
        num_samples = 2 * self.user_count * self.sample_size
        all_samples = np.random.choice(items, num_samples, p=probs)

        seen_samples = {}
        negative_samples = {}
        print('Sampling negative items by popularity...')
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

    def items_by_popularity(self):
        popularity = Counter()
        self.users = sorted(self.train.keys())
        for user in self.users:
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])

        popularity = dict(popularity)
        popularity = {k: v for k, v in sorted(popularity.items(), key=lambda item: item[1], reverse=True)}
        return popularity