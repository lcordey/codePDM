from numpy.lib.function_base import kaiser
import torch
import pickle
# import json
import yaml
import glob
import imageio
from skimage import color
import cv2
import matplotlib.pyplot as plt

from marching_cubes_rgb import *
from utils import *
import IPython

RANDOM_INIT = False
NUM_ITER = 10
DEFAULT_RENDER_RESOLUTION = 64
# DEFAULT_MAX_MODEL_2_RENDER = 4
DEFAULT_MAX_MODEL_2_RENDER = None
DEFAULT_IMAGES_PER_MODEL = 1

DECODER_PATH = "models_and_codes/decoder.pth"
ENCODER_PATH = "models_and_codes/encoderGrid.pth"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
PARAM_FILE = "config/param_encoder.yaml"
VEHICLE_VALIDATION_PATH = "config/vehicle_validation.txt"
MATRIX_PATH = "../../image2sdf/input_images_validation/matrix_w2c.pkl"
ANNOTATIONS_PATH = "../../image2sdf/input_images_validation/annotations.pkl"
IMAGES_PATH = "../../image2sdf/input_images_validation/images/"
OUTPUT_DIR = "../../image2sdf/encoder_output/evaluation"
LOGS_PATH = "../../image2sdf/logs/encoder/log.pkl"
PLOT_PATH = "../../image2sdf/plots/encoder/"

def init_xyz(resolution):
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz


def load_grid(list_hash, annotations, num_model_2_render, param_image, param_network):

    matrix_world_to_camera = pickle.load(open(MATRIX_PATH, 'rb'))

    num_model = len(list_hash)

    width_image = param_image["width"]
    height_image = param_image["height"]
    width_network = param_network["width"]
    height_network = param_network["height"]
    num_slices = param_network["num_slices"]

    
    all_grid = torch.empty([num_model, num_model_2_render, 3, num_slices, width_network, height_network], dtype=torch.float)

    list_id = list(annotations.keys())

    for model_hash, model_id in zip(list_hash, range(num_model)):
        for image_pth, image_id in zip(glob.glob(IMAGES_PATH + model_hash + '/*'), range(num_model_2_render)):

            # Load data and get label
            image_pth = IMAGES_PATH + model_hash + '/' + str(image_id) + '.png'
            input_im = imageio.imread(image_pth)

            loc_3d = annotations[model_hash][image_id]['3d'].copy()
            frame = annotations[model_hash][image_id]['frame'].copy()

            # interpolate slices vertex coordinates
            loc_slice_3d = np.empty([num_slices,4,3])
            for i in range(num_slices):
                loc_slice_3d[i,0,:] = loc_3d[0,:] * (1-i/(num_slices-1)) + loc_3d[4,:] * i/(num_slices-1)
                loc_slice_3d[i,1,:] = loc_3d[1,:] * (1-i/(num_slices-1)) + loc_3d[5,:] * i/(num_slices-1)
                loc_slice_3d[i,2,:] = loc_3d[2,:] * (1-i/(num_slices-1)) + loc_3d[6,:] * i/(num_slices-1)
                loc_slice_3d[i,3,:] = loc_3d[3,:] * (1-i/(num_slices-1)) + loc_3d[7,:] * i/(num_slices-1)

            # convert to image plane coordinate
            loc_slice_2d = np.empty_like(loc_slice_3d)
            for i in range(num_slices):
                for j in range(4):
                    loc_slice_2d[i,j,:] = convert_w2c(matrix_world_to_camera, frame, loc_slice_3d[i,j,:]) 

            ###### y coordinate is inverted + rescaling #####
            loc_slice_2d[:,:,1] = 1 - loc_slice_2d[:,:,1]
            loc_slice_2d[:,:,0] = loc_slice_2d[:,:,0] * width_image
            loc_slice_2d[:,:,1] = loc_slice_2d[:,:,1] * height_image

            # grid to give as input to the network
            input_grid = np.empty([num_slices, width_network, height_network, 3])


            # fill grid by slices
            for i in range(num_slices):
                src = loc_slice_2d[i,:,:2].copy()
                dst = np.array([[0, height_network], [width_network, height_network], [width_network, 0], [0,0]])
                h, mask = cv2.findHomography(src, dst)
                slice = cv2.warpPerspective(input_im, h, (width_network,height_network))
                input_grid[i,:,:,:] = slice

            # rearange, normalize and convert to tensor
            input_grid = np.transpose(input_grid, [3,0,1,2])
            input_grid = input_grid/255 - 0.5
            input_grid = torch.tensor(input_grid, dtype = torch.float)

            all_grid[model_id, image_id, :, :, :, :] = input_grid

    return all_grid


def get_code_from_grid(grid, latent_size):

    encoder = torch.load(ENCODER_PATH).cuda()
    encoder.eval()

    num_model = grid.shape[0]
    num_images_per_model = grid.shape[1]

    lat_code = torch.empty([num_model, num_images_per_model, latent_size]).cuda()
    for model_id in range(num_model):
        for image_id in range(num_images_per_model):
            lat_code[model_id, image_id, :] = encoder(grid[model_id, image_id, :, :, :, :].unsqueeze(0).cuda()).detach()


    return lat_code

# load parameters
param_all = yaml.safe_load(open(PARAM_FILE))
param = param_all["encoder"]

# load annotations
annotations = pickle.load(open(ANNOTATIONS_PATH, "rb"))
dict_hash_2_code = pickle.load(open(LATENT_CODE_PATH, 'rb'))

# Get validation model
with open(VEHICLE_VALIDATION_PATH) as f:
    list_hash_validation = f.read().splitlines()
list_hash_validation = list(list_hash_validation)

if DEFAULT_MAX_MODEL_2_RENDER is not None:
    list_hash_validation = list_hash_validation[:DEFAULT_MAX_MODEL_2_RENDER]

# only keep the ones for which with have annotated images
list_hash = []
for hash in list_hash_validation:
    if hash in annotations.keys():
        list_hash.append(hash)

num_model = len(list_hash)
num_images_per_model = len(annotations[list_hash[0]])
num_model_2_render= min(num_images_per_model, DEFAULT_IMAGES_PER_MODEL)

resolution = DEFAULT_RENDER_RESOLUTION

# compute latent codes
print("load grid...")
grid = load_grid(list_hash, annotations, num_model_2_render, param["image"], param["network"])
print("compute code from grid...")
latent_code = get_code_from_grid(grid, param_all["latent_size"])


# fill a xyz grid to give as input to the decoder 
xyz = init_xyz(resolution)



# load decoder
decoder = torch.load(DECODER_PATH).cuda()
decoder.eval()

# model_id = 0
# model_hash = list_hash[model_id]

print("start evaluation:\n")

for model_hash, model_id in zip(list_hash, range(num_model)):

    code_gt = dict_hash_2_code[model_hash].cuda()

    if RANDOM_INIT:
        code_prediction = torch.zeros(6).cuda()
    else:
        code_prediction = latent_code[model_id,:,:].mean(dim=0)

    # code_prediction.requires_grad = True

    # optimizer = torch.optim.Adam(
    # [
    #     {
    #         "params": code_prediction,
    #         "lr": 0.5,
    #         "eps": 1e-8,
    #     },
    # ]
    # )

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # for i in range(NUM_ITER):
    #     optimizer.zero_grad()

    #     sdf_pred = decoder(code_prediction.repeat(resolution * resolution * resolution, 1).cuda(),xyz)
    #     sdf_gt = decoder(code_gt.repeat(resolution * resolution * resolution, 1).cuda(),xyz)

    #     # assign weight of 0 for easy samples that are well trained
    #     threshold_precision = 1
    #     weight_sdf = ~((sdf_pred[:,0] > threshold_precision).squeeze() * (sdf_gt[:,0] > threshold_precision).squeeze()) \
    #         * ~((sdf_pred[:,0] < -threshold_precision).squeeze() * (sdf_gt[:,0] < -threshold_precision).squeeze())

    #     # loss l1 in distance error per samples
    #     loss_sdf = torch.nn.L1Loss(reduction='none')(sdf_pred[:,0].squeeze(), sdf_gt[:,0])
    #     loss_sdf = (loss_sdf * weight_sdf).mean() * weight_sdf.numel()/weight_sdf.count_nonzero()

    #     # loss rgb in pixel value difference per color per samples
    #     rgb_gt_normalized = sdf_gt[:,1:]
    #     loss_rgb = torch.nn.L1Loss(reduction='none')(sdf_pred[:,1:], rgb_gt_normalized)
    #     loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).mean()/3 * weight_sdf.numel()/weight_sdf.count_nonzero()


    #     total_loss = loss_sdf + loss_rgb

    #     total_loss.backward()
    #     optimizer.step()


    # print(f"\nmodel {model_id}:")
    # print(f"total loss: {total_loss}")
    # print(f"distance to the original code {(code_prediction - code_gt).norm().item()} ")



    sdf_pred = decoder(code_prediction.repeat(resolution * resolution * resolution, 1).cuda(),xyz)
    sdf_gt = decoder(code_gt.repeat(resolution * resolution * resolution, 1).cuda(),xyz)

    sdf_pred[:,1:] = sdf_pred[:,1:] * 255
    sdf_pred = sdf_pred.reshape(resolution, resolution, resolution, 4).cpu().detach().numpy()
    if(np.min(sdf_pred[:,:,:,0]) < 0 and np.max(sdf_pred[:,:,:,0]) > 0):
        vertices_pred, faces_pred = marching_cubes(sdf_pred[:,:,:,0])
        colors_v_pred = exctract_colors_v(vertices_pred, sdf_pred)

    vertices_pred = torch.tensor(vertices_pred.copy())
    colors_v_pred = torch.tensor(colors_v_pred/255).unsqueeze(0).cuda()


    sdf_gt[:,1:] = sdf_gt[:,1:] * 255
    sdf_gt = sdf_gt.reshape(resolution, resolution, resolution, 4).cpu().detach().numpy()
    if(np.min(sdf_gt[:,:,:,0]) < 0 and np.max(sdf_gt[:,:,:,0]) > 0):
        vertices_gt, faces_gt = marching_cubes(sdf_gt[:,:,:,0])
        colors_v_gt = exctract_colors_v(vertices_gt, sdf_gt)

    vertices_gt = torch.tensor(vertices_gt.copy())
    colors_v_gt = torch.tensor(colors_v_gt/255).unsqueeze(0).cuda()

    cham_sdf, cham_rgb, cham_lab = chamfer_distance_rgb(vertices_pred, vertices_gt, colors_x = colors_v_pred, colors_y = colors_v_gt)


    vertices_pred.requires_grad = True
    vertices_gt.requires_grad = True

    optimizer = torch.optim.Adam(
    [
        {
            "params": vertices_pred,
            "lr": 0.1,
        },

        {
            "params": colors_v_pred,
            "lr": 0.1,
        },
    ]
    )

    print(f"\nmodel {model_id}:")
    print(f"cham sdf init: {cham_sdf.item()}")

    for i in range(NUM_ITER):
        optimizer.zero_grad()

        cham_sdf, cham_rgb, cham_lab = chamfer_distance_rgb(vertices_pred, vertices_gt, colors_x = colors_v_pred, colors_y = colors_v_gt)

        # print(cham_rgb.item())

        cham_sdf.backward()
        # cham_rgb.backward()

        optimizer.step()
    
    print(f"cham sdf final: {cham_sdf.item()}")