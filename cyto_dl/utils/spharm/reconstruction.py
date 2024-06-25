from contextlib import suppress
from typing import Tuple

import numpy as np
import pandas as pd
import pyshtools
import vtk
from aicscytoparam import cytoparam
from vtk.util.numpy_support import vtk_to_numpy


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
    mesh, _ = get_reconstruction_from_coeffs(coeffs)
    return mesh


# the functions below have been taken from
# https://github.com/AllenCell/aics-shparam/blob/main/aicsshparam/shtools.py
def get_reconstruction_from_grid(grid: np.array, centroid: Tuple = (0, 0, 0)):
    """Converts a parametric 2D grid of type (lon,lat,rad) into a 3d mesh. lon in [0,2pi], lat in.

    [0,pi].

    Parameters
    ----------
    grid : np.array
        Input grid where the element grid[i,j] represents the
        radial coordinate at longitude i*2pi/grid.shape[0] and
        latitude j*pi/grid.shape[1].

    Returns
    -------
    mesh : vtkPolyData
        Mesh that represents the input parametric grid.

    Other parameters
    ----------------
    centroid : tuple of floats, optional
        x, y and z coordinates of the centroid where the mesh
        will be translated to, default is (0,0,0).
    """

    res_lat = grid.shape[0]
    res_lon = grid.shape[1]

    # Creates an initial spherical mesh with right dimensions.
    rec = vtk.vtkSphereSource()
    rec.SetPhiResolution(res_lat + 2)
    rec.SetThetaResolution(res_lon)
    rec.Update()
    rec = rec.GetOutput()

    grid_ = grid.T.flatten()

    # Update the points coordinates of the spherical mesh according to the input grid
    for j, lon in enumerate(np.linspace(0, 2 * np.pi, num=res_lon, endpoint=False)):
        for i, lat in enumerate(
            np.linspace(np.pi / (res_lat + 1), np.pi, num=res_lat, endpoint=False)
        ):
            theta = lat
            phi = lon - np.pi
            k = j * res_lat + i
            x = centroid[0] + grid_[k] * np.sin(theta) * np.cos(phi)
            y = centroid[1] + grid_[k] * np.sin(theta) * np.sin(phi)
            z = centroid[2] + grid_[k] * np.cos(theta)
            rec.GetPoints().SetPoint(k + 2, x, y, z)
    # Update coordinates of north and south pole points
    north = grid_[::res_lat].mean()
    south = grid_[(res_lat - 1) :: res_lat].mean()
    rec.GetPoints().SetPoint(0, centroid[0] + 0, centroid[1] + 0, centroid[2] + north)
    rec.GetPoints().SetPoint(1, centroid[0] + 0, centroid[1] + 0, centroid[2] - south)

    # Compute normal vectors
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(rec)
    # Set splitting off to avoid output mesh from having different number of
    # points compared to input
    normals.SplittingOff()
    normals.Update()

    mesh = normals.GetOutput()

    return mesh


def get_reconstruction_from_coeffs(coeffs: np.array, lrec: int = 0):
    """Converts a set of spherical harmonic coefficients into a 3d mesh.

    Parameters
    ----------
    coeffs : np.array
        Input array of spherical harmonic coefficients. These
        array has dimensions 2xLxM, where the first dimension
        is 0 for cosine-associated coefficients and 1 for
        sine-associated coefficients. Second and third dimensions
        represent the expansion parameters (l,m).

    Returns
    -------
    mesh : vtkPolyData
        Mesh that represents the input parametric grid.

    Other parameters
    ----------------
    lrec : int, optional
        Degree of the reconstruction. If lrec<l, then only
        coefficients l<lrec will be used for creating the mesh.
        If lrec>l, then the mesh will be oversampled.
        Default is 0 meaning all coefficients
        available in the matrix coefficients will be used.

    Notes
    -----
        The mesh resolution is set by the size of the coefficients
        matrix and therefore not affected by lrec.
    """

    # Degree of the expansion
    lmax = coeffs.shape[1]

    if lrec == 0:
        lrec = lmax

    # Create array (oversampled if lrec>lrec)
    coeffs_ = np.zeros((2, lrec, lrec), dtype=np.float32)

    # Adjust lrec to the expansion degree
    if lrec > lmax:
        lrec = lmax

    # Copy coefficients
    coeffs_[:, :lrec, :lrec] = coeffs[:, :lrec, :lrec]

    # Expand into a grid
    grid = pyshtools.expand.MakeGridDH(coeffs_, sampling=2)

    # Get mesh
    mesh = get_reconstruction_from_grid(grid)

    return mesh, grid
