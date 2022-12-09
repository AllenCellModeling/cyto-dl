from monai.transforms import Transform


class EmbedSegConcatLabelsd(Transform):
    def __init__(self, input_keys, output_key):
        super().__init__()
        self.input_keys = input_keys
        self.output_key = output_key

    def __call__(self, image_dict):
        print(image_dict.keys())
        image_dict[self.output_key] = {}
        for key in self.input_keys:
            image_dict[self.output_key][key] = image_dict[key].as_tensor()
            del image_dict[key]
        return image_dict