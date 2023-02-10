from contextlib import suppress

import numpy as np
import pandas as pd
import vtk
from aicscytoparam import cytoparam
from aicsshparam import shtools
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def get_image_from_shcoeffs(x, spharm_cols_filter, spharm_cols):
    x = pd.DataFrame(x).T
    x.columns = spharm_cols
    x = x.iloc[0]

    mesh = get_mesh_from_series(x, spharm_cols_filter["startswith"][:-1], 32)
    # Find mesh coordinates
    coords = vtk_to_numpy(mesh.GetPoints().GetData())

    # Find bounds of the mesh
    rmin = (coords.min(axis=0) - 0.5).astype(np.int)
    rmax = (coords.max(axis=0) + 0.5).astype(np.int)
    # Create image data
    imagedata = vtk.vtkImageData()
    w, h, d = 150, 150, 150
    imagedata.SetDimensions([w, h, d])
    imagedata.SetExtent(0, w - 1, 0, h - 1, 0, d - 1)
    imagedata.SetOrigin(rmin)
    imagedata.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    # Set all values to 1
    imagedata.GetPointData().GetScalars().FillComponent(0, 1)

    # Create an empty 3D numpy array to sum up
    # voxelization of all meshes
    img = np.zeros((d, h, w), dtype=np.uint8)

    # Voxelize mesh
    seg = cytoparam.voxelize_mesh(imagedata=imagedata, shape=(d, h, w), mesh=mesh, origin=rmin)
    img[seg > 0] = 1
    return img


def get_mesh_from_series(row, alias, lmax):
    coeffs = np.zeros((2, lmax, lmax), dtype=np.float32)
    for l in range(lmax):  # noqa E741
        for m in range(l + 1):
            # If a given (l,m) pair is not found, it is assumed to be zero
            with suppress(IndexError):
                # Cosine SHE coefficients
                coeffs[0, l, m] = row[
                    [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}C" in f]
                ]
                # Sine SHE coefficients
                coeffs[1, l, m] = row[
                    [f for f in row.keys() if f"{alias}_shcoeffs_L{l}M{m}S" in f]
                ]
    mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs)
    return mesh
