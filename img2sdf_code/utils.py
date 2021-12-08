import numpy as np
import torch

from pytorch3d.ops.knn import knn_points
from skimage import color

def chamfer_distance_rgb(
    x,
    y,
    colors_x = None,
    colors_y = None
):

    """
    Chamfer distance between two pointclouds x and y.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! Only support batch of size 1 !!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].

    Returns:
        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
    """

    # checkt point inputs size
    if x.ndim == 3 and y.ndim == 3 and x.shape[0] == 1 and y.shape[0] == 1:
        pass
    elif x.ndim == 2 and y.ndim == 2:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    else:
        raise ValueError("Expected points to be of shape (1, P, D) or (P,D)")
    

    if colors_x is not None and colors_y is not None:
        return_cham_colors = True
    else:
        return_cham_colors = False
    
    # get dimension
    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")


    # checkt colors inputs size
    if return_cham_colors:
        if colors_x.ndim == 3 and colors_y.ndim == 3 and colors_x.shape[0] == 1 and colors_y.shape[0] == 1:
            colors_x = colors_x.squeeze()
            colors_y = colors_y.squeeze()
        elif colors_x.ndim == 2 and colors_y.ndim == 2:
            pass
        else:
            raise ValueError("Expected colors of points to be of shape (1, P, D) or (P,D)")

        if colors_x.shape[0] != P1 or colors_y.shape[0] != P2:
            raise ValueError("colors inputs size must match points size.")

        if colors_x.shape[1] != 3 or colors_y.shape[1] != 3:
            raise ValueError("last dimension or colors input should be of size 3 as it should be rgb values")

        if colors_x.max() > 1.0 or colors_x.min() < 0.0 or colors_y.max() > 1.0 or colors_y.min() < 0.0:
            raise ValueError("Colors values must be normalized between 0 and 1")
            


    # find nearest neighbours
    x_nn = knn_points(x, y, K=1)
    y_nn = knn_points(y, x, K=1)

    # get distance error from knn
    cham_x = x_nn.dists[..., 0]
    cham_y = y_nn.dists[..., 0]

    cham_x = cham_x.squeeze().sum()
    cham_y = cham_y.squeeze().sum()

    # average through all points
    cham_x = cham_x/P1
    cham_y = cham_y/P2

    cham_dist = cham_x + cham_y


    if return_cham_colors:

        # # get lab colors too
        # colors_x_lab = torch.tensor(color.rgb2lab(colors_x.cpu().numpy())).cuda()
        # colors_y_lab = torch.tensor(color.rgb2lab(colors_y.cpu().numpy())).cuda()
        
        # find index from knn results
        idx_x = x_nn.idx[..., 0].squeeze()
        idx_y = y_nn.idx[..., 0].squeeze()

        # compute rgb error
        error_x_rgb = torch.nn.L1Loss()(colors_x[:], colors_y[idx_x])
        error_y_rgb = torch.nn.L1Loss()(colors_y[:], colors_x[idx_y])

        # normalize
        cham_colors_rgb = (error_x_rgb + error_y_rgb) / 2
        cham_colors_rgb = cham_colors_rgb * 255

        # # compute lab error
        # error_x_lab = torch.nn.L1Loss()(colors_x_lab[:], colors_y_lab[idx_x])
        # error_y_lab = torch.nn.L1Loss()(colors_y_lab[:], colors_x_lab[idx_y])

        # cham_colors_lab = (error_x_lab + error_y_lab) / 2

        cham_colors_lab = None
        
    else:
        cham_colors_rgb = None
        cham_colors_lab = None

    return cham_dist, cham_colors_rgb, cham_colors_lab



def convert_w2c(matrix_world_to_camera, frame, point):

    point_4d = np.resize(point, 4)
    point_4d[3] = 1
    co_local = matrix_world_to_camera.dot(point_4d)
    z = -co_local[2]

    f = np.empty([3,3])

    if z == 0.0:
        return np.array([0.5, 0.5, 0.0])
    else:
        for i in range(3):
            f[i] =  -(frame[i] / (frame[i][2]/z))

    min_x, max_x = f[2][0], f[1][0]
    min_y, max_y = f[1][1], f[0][1]

    x = (co_local[0] - min_x) / (max_x - min_x)
    y = (co_local[1] - min_y) / (max_y - min_y)

    return np.array([x,y,z])


def convert_view_to_camera_coordinates(frame, pixel_location):

    x = pixel_location[0]
    y = pixel_location[1]

    f = np.empty([3,3])
    camera_coordinate = np.empty([4])
    camera_coordinate[2] = -1 # z = 1
    camera_coordinate[3] = 1


    for i in range(3):
        f[i] =  -(frame[i] / (frame[i][2]))

    min_x, max_x = f[2][0], f[1][0]
    min_y, max_y = f[1][1], f[0][1]

    camera_coordinate[0] = x * (max_x - min_x) + min_x
    camera_coordinate[1] = y * (max_y - min_y) + min_y

    return camera_coordinate

