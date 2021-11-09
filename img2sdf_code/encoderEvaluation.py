import torch
import pickle
import json
import glob
import imageio
import cv2
import matplotlib.pyplot as plt

from marching_cubes_rgb import *
import IPython

DEFAULT_RENDER = False
DEFAULT_RENDER_RESOLUTION = 64
DEFAULT_MAX_MODEL_2_RENDER = 3
DEFAULT_LOGS = True

DECODER_PATH = "models_and_codes/decoder.pth"
ENCODER_PATH = "models_and_codes/encoderGrid.pth"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
# PARAM_FILE = "config/param.json"
PARAM_FILE = "config/param.yaml"
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
        dict_hash_2_code = pickle.load(open(LATENT_CODE_PATH, 'rb'))

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
        num_images_per_model = len(annotations[list_hash[0]])
        num_model_2_render= min(num_images_per_model, args.num_image)

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



            # decode
            sdf_target = np.empty([resolution, resolution, resolution, 4])

            for x in range(resolution):

                sdf_pred = decoder(dict_hash_2_code[model_hash].repeat(resolution * resolution, 1),xyz[x * resolution * resolution: (x+1) * resolution * resolution]).detach()

                sdf_pred[:,0] = sdf_pred[:,0] * resolution
                sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
                sdf_pred[:,1:] = sdf_pred[:,1:] * 255

                sdf_target[x, :, :, :] = np.reshape(sdf_pred[:,:].cpu(), [resolution, resolution, 4])

            # print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_target[:,:,:,0]), np.max(sdf_target[:,:,:,0])))
            if(np.min(sdf_target[:,:,:,0]) < 0 and np.max(sdf_target[:,:,:,0]) > 0):
                vertices, faces = marching_cubes(sdf_target[:,:,:,0])
                colors_v = exctract_colors_v(vertices, sdf_target)
                colors_f = exctract_colors_f(colors_v, faces)
                off_file = "%s/%s_gt.off" %(OUTPUT_DIR, model_hash)
                write_off(off_file, vertices, faces, colors_f)
                print("Wrote %s_gt.off" % model_hash)
            else:
                print("surface level: 0, should be comprise in between the minimum and maximum value")

            # compute the sdf from codes 
            sdf_validation = torch.tensor(sdf_result).reshape(resolution * resolution * resolution, 4)
            sdf_target= torch.tensor(sdf_target).reshape(resolution * resolution * resolution, 4)

            # assign weight of 0 for easy samples that are well trained
            threshold_precision = 1/resolution
            weight_sdf = ~((sdf_validation[:,0] > threshold_precision).squeeze() * (sdf_target[:,0] > threshold_precision).squeeze()) \
                * ~((sdf_validation[:,0] < -threshold_precision).squeeze() * (sdf_target[:,0] < -threshold_precision).squeeze())

            # loss l1 in distance error per samples
            loss_sdf = torch.nn.L1Loss(reduction='none')(sdf_validation[:,0].squeeze(), sdf_target[:,0])
            loss_sdf = (loss_sdf * weight_sdf).mean() * weight_sdf.numel()/weight_sdf.count_nonzero()
        
            # loss rgb in pixel value difference per color per samples
            rgb_gt_normalized = sdf_target[:,1:]
            loss_rgb = torch.nn.L1Loss(reduction='none')(sdf_validation[:,1:], rgb_gt_normalized)
            loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).mean()/3 * weight_sdf.numel()/weight_sdf.count_nonzero()

            print(f"loss_sdf: {loss_sdf}")
            print(f"loss_rgb: {loss_rgb}")


    if args.logs:

        # load parameters
        logs = pickle.load(open(LOGS_PATH, 'rb'))

        param_all = json.load(open(PARAM_FILE))
        param = param_all["encoder"]

        # num_batch_per_epoch = num_model * num_images_per_model / param["dataLoader"]["batch_size"]
        num_batch_per_epoch = 688 * 300 / param["dataLoader"]["batch_size"]
        num_print_per_epoch = num_batch_per_epoch / param["num_batch_between_print"]
        num_validation_per_epoch = num_batch_per_epoch / param["num_batch_between_validation"]
        x_timestamp_training = np.arange(len(logs["training"])) / num_print_per_epoch 
        x_timestamp_validation = np.arange(len(logs["validation"]["l2"])) / (num_validation_per_epoch + 1)
        x_timestamp_validation_temp = np.arange(len(logs["validation"]["sdf"])) / (num_validation_per_epoch + 1)

        # let's plots :)
        # sdf
        plt.figure()
        plt.title("logs loss training")
        plt.semilogy(x_timestamp_training,logs["training"])
        plt.ylabel("loss l2")
        plt.xlabel("epoch")
        plt.savefig(PLOT_PATH + "training.png")


        plt.figure()
        plt.title("logs loss l2 validation")
        plt.semilogy(x_timestamp_validation,logs["validation"]["l2"])
        plt.ylabel("loss l2")
        plt.xlabel("epoch")
        plt.savefig(PLOT_PATH + "l2_val.png")

        plt.figure()
        plt.title("logs loss sdf validation")
        plt.semilogy(x_timestamp_validation_temp,logs["validation"]["sdf"])
        plt.ylabel("error sdf")
        plt.xlabel("epoch")
        plt.savefig(PLOT_PATH + "sdf_val.png")

        plt.figure()
        plt.title("logs loss rgb validation")
        plt.semilogy(x_timestamp_validation_temp,logs["validation"]["rgb"])
        plt.ylabel("error rgb")
        plt.xlabel("epoch")
        plt.savefig(PLOT_PATH + "rgb_val.png")

    print("done")


