import math
import random
from collections import deque

import torch


def window(iterable, chunk_size=3, overlap=0):
    if chunk_size < 1:
        raise Exception("chunk size too small")
    if overlap >= chunk_size:
        raise Exception("overlap too large")
    queue = deque(maxlen=chunk_size)
    it = iter(iterable)
    i = 0
    try:
        # start by filling the queue with the first group
        for i in range(chunk_size):
            queue.append(next(it))
        while True:
            yield tuple(queue)
            # after yielding a chunk, get enough elements for the next chunk
            for i in range(chunk_size - overlap):
                queue.append(next(it))
    except StopIteration:
        # if the iterator is exhausted, yield any remaining elements
        i += overlap
        if i > 0:
            yield tuple(queue)[-i:]


class TrackSampler(torch.utils.data.Sampler):
    def __init__(self, track_indices, batch_size, overlap=None):
        self.track_indices = track_indices
        self.batch_size = batch_size

        if overlap is None:
            overlap = batch_size // 2
        self.overlap = overlap

        self.total_batches = 0
        for track in track_indices:
            self.total_batches += math.ceil((len(track) - overlap) / (batch_size - overlap))

    def __iter__(self):
        iterators = [
            window(track, chunk_size=self.batch_size, overlap=self.overlap)
            for track in self.track_indices
        ]

        iterator_indices = list(range(len(iterators)))

        for _ in range(self.total_batches):
            while True:
                track_it_idx = random.choice(iterator_indices)  # nosec
                track_it = iterators[track_it_idx]

                try:
                    yield next(track_it)
                    break
                except StopIteration:
                    iterators.pop(track_it_idx)
                    iterator_indices = list(range(len(iterators)))

    def __len__(self):
        return self.total_batches
