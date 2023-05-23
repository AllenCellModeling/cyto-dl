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


def random_chunks(iterable, chunk_size, n_chunks):
    last_possible_start = len(iterable) - chunk_size - 1
    for chunk in range(n_chunks):
        start_ix = random.randint(0, last_possible_start)  # nosec
        yield iterable[start_ix : start_ix + chunk_size]


class TrackSampler(torch.utils.data.Sampler):
    def __init__(self, track_indices, batch_size, overlap=None, n_random_chunks=None):
        self.track_indices = track_indices
        self.batch_size = batch_size

        self.total_batches = 0
        if overlap is not None:
            self.overlap = overlap
            self.n_random_chunks = None
            for track in track_indices:
                self.total_batches += math.ceil((len(track) - overlap) / (batch_size - overlap))
        else:
            self.overlap = None
            if n_random_chunks is None:
                n_random_chunks = 2
            self.n_random_chunks = n_random_chunks

            self.total_batches = len(track_indices) * n_random_chunks

    def __iter__(self):
        if self.overlap is not None:
            iterators = [
                window(track, chunk_size=self.batch_size, overlap=self.overlap)
                for track in self.track_indices
            ]
        else:
            iterators = [
                random_chunks(track, chunk_size=self.batch_size, n_chunks=self.n_random_chunks)
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
