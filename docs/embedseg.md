## EmbedSeg Docs

Embedseg is a an instance segmentation method described [here](https://juglab.github.io/EmbedSeg/). Here we will go over the parts of the config that are uniquely important for EmbedSeg.

#### Data Config

The inputs to the datamodule is a csv with columns of filepaths to an unnormalized 3D microscopy image (called `raw` in the example config) and its corresponding instance segmentation (called `seg` in the example) and a `split` column that marks whether each row is used for `train`, `test`, or `val`.

The data config details the preprocessing to transform this csv into images usable for EmbedSeg training.

```
transforms:
  train:
    _target_: monai.transforms.Compose
    transforms:
```

This section denotes the transforms performed during training

```
- _target_: monai.transforms.LoadImaged
        keys: ['raw', 'seg']
        reader:
        - _target_: aics_im2im.image.io.MonaiBioReader
          dimension_order_out: 'CZYX'
```

Here, we load the images pointed to by the `raw` and `seg` columns in CZYX order.

```
- _target_: monai.transforms.ToTensord
        keys: ['raw', 'seg']
```

This converts the loaded images to tensors.

```
- _target_: monai.transforms.NormalizeIntensityd
        keys: ['raw']
```

This z-score whitens the `raw` image

```
- _target_: aics_im2im.utils.ExtractCentroidd
        input_key: 'seg'
        output_key:'CE'
```

The `ExtractCentroidd` transform takes the instance segmentation and creates an additional image, `CE`, which is an image with the centroid positions of `seg`.

```
- _target_: aics_im2im.image.transforms.RandomMultiScaleCropd
        keys: ['raw', 'seg',  'CE']
        patch_shape: ${model.patch_shape}
        patch_per_image: 1
        scales_dict:
          1: ['raw', 'seg',  'CE']
        selection_fn:
          _target_: aics_im2im.image.transforms.bright_sampler.AnySampler
          key: CE
          threshold: 1
          base_prob: 0
```

This transform randomly crops 1 patch from the larger `raw`, `seg`, `CL`, and `CE` images. Optionally, we can use a `selection_fn`, which filters random proposal crops based on whether the `selection_fn` evaluates to `True` when passed the `CE` image. Here, we use `BrightSampler` with a threshold of 0 on the centroid key `CE`, meaning that we only accept random crops that contain at least 1 centroid.

```
- _target_: aics_im2im.utils.EmbedSegConcatLabelsd
        input_keys: ['seg', 'CE']
        output_key: GT
```

Finally, we combine the `seg` and `CE` images into a dictionary called `GT`. This is required for calculating the EmbedSeg loss.

Validation uses all of the same transforms, while testing and prediction only use the predictions involving the `raw` image.

#### Model Config

1. Backbone architecture
   For a simple, single-task EmbedSeg Network, we output 7 channels (this is based on $2\*`n_sigma` + 1$).
2. Tasks

- The name of this task should be `GT` to align it with the `output_key` from the `EmbedSegConcatLabelsd` transform. We do not need extra postprocessing from the backbone in the simple, single-task case so we can use the `IdentityAuxHead`.
- The EmbedSeg Loss should use:
  - `grid` sizes that match your patch_shape.
  - `pixel` should line up with the approximate anisotropy ratios of your input images.
  - `n_sigma` is used to determine how the output of the backbone is interpreted and can be left at 3.
  - `foreground_weight` is based on the ratio of foreground pixels to background pixels.
  - `use_costmap` is currently not supported and should be left as False
  - `instance_key` and `center_key` refer to the input keys in `EmbedSegConcatLabelsd`

3. Postprocessing
   A clustering step is necessary for visualization of model predictions during prediction. The `pixel` and `n-sigma` arguments should match those used for the `SpatialEmbLoss_3d`.
