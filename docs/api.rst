============
Using the API
============

CytoDL can be imported as a Python package and used programatically with the `cyto_dl.api.CytoDLModel` class in addition to the command line interface. We provide utilities for loading configuration files, training models, and making predictions.

+++++++++++++++
Loading Configs
+++++++++++++++
Configs can be loaded from a python dictionary
    ```python   
    from cyto_dl.api import CytoDLModel
    cfg = {
        'data':{...},
        'model':{...},
        ...
    }
    model = CytoDLModel()
    model.load_config_from_dict(cfg)
    ```
or from a yaml file
    ```python
    from cyto_dl.api import CytoDLModel
    model = CytoDLModel()
    model.load_config_from_yaml('path/to/config.yaml')
    ```
or by name from one of our default configs. Available options are 'gan', 'instance_seg', 'labelfree', and 'segmenation'. When loading a default experiment, overrides can be provided as a list of strings formatted in the same way as for the CLI. 
    ```python
    from cyto_dl.api import CytoDLModel
    model = CytoDLModel()
    model.load_default_experiment('gan', overrides = ['data.batch_size=16'])
    ```

Once a config is loaded, attributes can be overriden using `model.override_config()`, which takes in a dictionary of values to be overriden. 
    ```python
    from cyto_dl.api import CytoDLModel
    model = CytoDLModel()
    model.load_default_experiment('gan')
    overrides = {'data.batch_size': 16}
    model.override_config(overrides)
    ```

+++++++++++++++
Training and Prediction
+++++++++++++++
Once a config is loaded, the model can be trained using `model.train()`.
    ```python
    from cyto_dl.api import CytoDLModel
    model = CytoDLModel()
    model.load_default_experiment('gan')
    await model.train()
    ```
The model can be used to make predictions using `model.predict()` 
    ```python
    from cyto_dl.api import CytoDLModel
    model = CytoDLModel()
    model.load_default_experiment('gan')
    await model.predict()
    ```

+++++++++++++++
Utils
+++++++++++++++
To examine the loaded config, use `model.print_config()`
    ```python
    from cyto_dl.api import CytoDLModel
    model = CytoDLModel()
    model.load_default_experiment('gan')
    model.print_config()
    ```
    
To download example data, use `model.download_example_data()`. This is useful when using the default models.
    ```python
    from cyto_dl.api import CytoDLModel
    model = CytoDLModel()
    model.download_example_data()
    ```