class BrightSampler:
    def __init__(self, key, threshold):
        self.key = key
        self.threshold = threshold

    def  __call__(self, patch):
        return patch[self.key]>self.threshold:

    