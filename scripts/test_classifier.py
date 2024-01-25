
from cyto_dl.datamodules.track_classification import TrackClassificationDatamodule
d = TrackClassificationDatamodule(csv_path='//allen/aics/assay-dev/users/Benji/CurrentProjects/breakdown_classifier_cyto_dl/data/', augmentations='')
d.prepare_data()
