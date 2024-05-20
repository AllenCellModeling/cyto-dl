import numpy as np
from skimage.draw import random_shapes
from skimage.measure import find_contours, label

from cyto_dl.image.io import PolygonLoaderd


def test_load_polygon(tmp_path):
    # create random image and find contours
    image = random_shapes((500, 500), max_shapes=1, num_channels=1, random_seed=3)[0].squeeze()
    # create polygon from background
    image = image == 255
    contours = find_contours(label(image))

    # Convert contours to polygons and save
    polygons = [contour.astype(int) for contour in contours]
    temp_file = tmp_path / "test.npy"
    np.save(temp_file, polygons)

    data = {
        "image": image,
        "poly": temp_file,
    }
    transform = PolygonLoaderd(keys=["poly"], shape_reference_key="image")
    reconstructed = transform(data)["poly"][0]

    # Check that the reconstructed mask is the close to as the original
    iou = np.sum(np.logical_and(reconstructed[0] > 0, image > 0)) / np.sum(
        np.logical_or(reconstructed[0] > 0, image > 0)
    )

    assert iou > 0.8

    # check that all slices are the same
    assert np.all([reconstructed[i] == reconstructed[0] for i in range(1, reconstructed.shape[0])])
