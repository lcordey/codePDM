import torch
import pickle
import json
import glob
import imageio
import cv2
import matplotlib.pyplot as plt

from marching_cubes_rgb import *
import IPython

DEFAULT_RENDER = True
DEFAULT_RENDER_RESOLUTION = 64
DEFAULT_MAX_MODEL_2_RENDER = 3
DEFAULT_LOGS = True

DECODER_PATH = "models_and_codes/decoder.pth"
ENCODER_PATH = "models_and_codes/encoderGrid.pth"
PARAM_FILE = "config/param.json"
VEHICLE_VALIDATION_PATH = "config/vehicle_validation.txt"
MATRIX_PATH = "../../image2sdf/input_images/matrix_w2c.pkl"
ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
IMAGES_PATH = "../../image2sdf/input_images/images/"
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


def convert_w2c(matrix_world_to_camera, frame, point):

    point_4d = np.resize(point, 4)
    point_4d[3] = 1
    co_local = matrix_world_to_camera.dot(point_4d)
    z = -co_local[2]

    if z == 0.0:
            return np.array([0.5, 0.5, 0.0])
    else:
        for i in range(3):
            frame[i] =  -(frame[i] / (frame[i][2]/z))

    min_x, max_x = frame[2][0], frame[1][0]
    min_y, max_y = frame[1][1], frame[0][1]

    x = (co_local[0] - min_x) / (max_x - min_x)
    y = (co_local[1] - min_y) / (max_y - min_y)

    return np.array([x,y,z])


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


if __name__ == '__main__':

    # read arguments
    parser = argparse.ArgumentParser(description='Render decoder results')
    parser.add_argument('--render', type=bool, help='render model -> True or False', default= DEFAULT_RENDER)
    parser.add_argument('--resolution', type=int, help='resolution -> int', default= DEFAULT_RENDER_RESOLUTION)
    parser.add_argument('--num_image', type=int, help='number of images to evaluate -> int', default= DEFAULT_MAX_MODEL_2_RENDER)
    parser.add_argument('--logs', type=bool, help='plots logs -> True or False', default= DEFAULT_LOGS)
    args = parser.parse_args()


    if args.render:

        # load parameters
        param_all = json.load(open(PARAM_FILE))
        param = param_all["encoder"]

        # load annotations
        annotations = pickle.load(open(ANNOTATIONS_PATH, "rb"))

        # Get validation model
        with open(VEHICLE_VALIDATION_PATH) as f:
            list_hash_validation = f.read().splitlines()
        list_hash_validation = list(list_hash_validation)

        # only keep the ones for which with have annotated images
        list_hash = []
        for hash in list_hash_validation:
            if hash in annotations.keys():
                list_hash.append(hash)

        num_model = len(list_hash)
        num_model_2_render= min(num_model, args.num_image)

        resolution = args.resolution

        # compute latent codes
        grid = load_grid(list_hash, annotations, num_model_2_render, param["image"], param["network"])
        latent_code = get_code_from_grid(grid, param_all["latent_size"])

        
        # fill a xyz grid to give as input to the decoder 
        xyz = init_xyz(resolution)

        # load decoder
        decoder = torch.load(DECODER_PATH).cuda()
        decoder.eval()

        for model_hash, model_id in zip(list_hash, range(num_model)):
            for j in range(num_model_2_render):
            
                # decode
                sdf_result = np.empty([resolution, resolution, resolution, 4])

                for x in range(resolution):

                    sdf_pred = decoder(latent_code[model_id,j,:].repeat(resolution * resolution, 1),xyz[x * resolution * resolution: (x+1) * resolution * resolution]).detach()

                    sdf_pred[:,0] = sdf_pred[:,0] * resolution
                    sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
                    sdf_pred[:,1:] = sdf_pred[:,1:] * 255

                    sdf_result[x, :, :, :] = np.reshape(sdf_pred[:,:].cpu(), [resolution, resolution, 4])


                # print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result[:,:,:,0]), np.max(sdf_result[:,:,:,0])))
                if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
                    vertices, faces = marching_cubes(sdf_result[:,:,:,0])
                    colors_v = exctract_colors_v(vertices, sdf_result)
                    colors_f = exctract_colors_f(colors_v, faces)
                    off_file = '%s/%s_%d.off' %(OUTPUT_DIR, model_hash, j)
                    write_off(off_file, vertices, faces, colors_f)
                    print('Wrote %s_%d.off' % (model_hash, j))
                else:
                    print("surface level: 0, should be comprise in between the minimum and maximum value")



    if args.logs:

        # load parameters
        logs = pickle.load(open(LOGS_PATH, 'rb'))

    print("done")


