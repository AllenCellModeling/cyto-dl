.. highlight:: shell

============
Modifying the Example Configs
============

Here we will outline some of the basic modifications that can be made to the provided example configurations if the default configurations does not work on your data.
We assume familiarity with running the default configurations and the `lightning-hydra-template` style.

1. Changes to the `data` config
Our data configs all follow the same structure - image loading, image normalization, and image augmentation. Any of these steps can be modified to better suit your needs.
    a. Image loading
        By default, we use :ref:aicsimageio`https://github.com/AllenCellModeling/aicsimageio`, which allows flexible image loading of many formats. In the examples, we assume 3D data and load ZYX slices from a multi-channel image.
        If your images are 2D, you can change `dimension_order_out` to `YX`. Note that the image processing transforms expect channel-first images.
    b. Image normalization
        It is recommended that you transform your images from raw pixel values into a format suitable for your use case. :ref:Monai`https://docs.monai.io/en/stable/transforms.html#intensity-dict` provides many useful transforms for this. Note that we use the dictionary versions of these transforms.
        Ground truth images should also be normalized based on your use case, e.g. binarized for segmentation, transformed to range [-1, 1] for GANs, etc.
    c. Image augmentation
        Targeted image augmentation can increase model robustness. Again, monai provides excellent options for :ref:intensity`https://docs.monai.io/en/stable/transforms.html#intensity-dict` and :ref:spatial`https://docs.monai.io/en/stable/transforms.html#spatial-dict` augmentations.
        For spatial augmentations, ensure that your input and ground truth images are both passed to the transformation, while for intensity augmentations ensure that only the input image is changed.
        **Note** For omnipose, do not spatially augment the generated Omnipose ground truth after the `OmniposePreprocessd` transform, as this will result in incorrect gradients in the ground truth.

2. Changes to the `model` config
The model config specifies neural network architecture and optimization parameters. Our templates provided a shared `backbone` with task-specific `task_heads`, for multi-task learning, (e.g. segmentation and labelfree signal prediction).
    a. Modifying the `backbone`
        :ref:monai`https://docs.monai.io/en/stable/networks.html#nets` provides many cutting edge networks. Crucial parameters to change are the `spatial_dims` if you are changing from a 3D to 2D task, `in_channels` if you want to provide multi-channel images to the network, and `out_channels`.
        For multi-task learning, it is important to increase the number of `out_channels` so that the task heads are not bottlenecked by the number of `out_channels` in the backbone.
    b. Modifying the `task_heads`
        `task_heads` can be modified by changing their loss function (suggested if you are changing e.g. from labelfree to segmentation), postprocessing (if you are changing from segmentation to omnipose), and `task-head` type (if you are changing from a segmentation network to a GAN).
        :ref:torch`https://pytorch.org/docs/stable/nn.html#loss-functions` and :ref:monai`https://docs.monai.io/en/stable/losses.html` provide many loss functions. We provide basic :ref:postprocessing`aics_im2im/models/utils/postprocessing` functions.
        Additional `task_heads` can be added for multi-task learning. The name of each `task_head` should line up with the name of an image in your training batch. For example, if our batch looks like `{'raw':torch.Tensor, 'segmentation':torch.Tensor, 'distance':torch.Tensor}` and `raw` is our input image,
        we should provide `task_heads`  for `segmentation` and `distance` that predict a segmentation and distance map respectively.


### Memory considerations
GPU memory is often a limiting factor in neural network training. Here, GPU memory use is primarily determined by combination of `model.patch_shape` x `data.batch_size`. As a rule of thumb, the `model.patch_shape` should be large enough to contain one instance of the entity that you are trying to predict.
Once `model.patch_size` is established, `data.batch_size` can be increased until GPU memory is exceeded. Alternatively, if 1 patch is too large for GPU memory, the :ref:Resized Transform`aics_im2im/image/transforms/resized.py` can be used to downsample images for training. Model size can also be decreased to decrease GPU memory usage.
