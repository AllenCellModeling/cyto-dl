from monai.transforms import Transform

class AddChannel(Transform):
    """
    This transform adds a channel dimension to the data.
    """
    def __init__(self, keys):
        """
        Args:
            keys (list of str): Keys to pick data for transformation.
        """
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            data[key] = img.unsqueeze(0)
        return data