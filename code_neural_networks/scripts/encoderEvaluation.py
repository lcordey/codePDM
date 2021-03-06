""" 
Evaluate the encoder
Render meshes and plots
print the shape(sdf), appearance(rgb and lab) loss and chamfer distance
"""

from numpy.lib.function_base import kaiser
import torch
import pickle
import yaml
import glob
import imageio
from skimage import color
import cv2
import matplotlib.pyplot as plt

from marching_cubes_rgb import *
from utils import *
import IPython

DEFAULT_RENDER = True
DEFAULT_RENDER_RESOLUTION = 64
DEFAULT_MAX_MODEL_2_RENDER = None
DEFAULT_IMAGES_PER_MODEL = 1
DEFAULT_LOGS = True

DECODER_PATH = "../models_and_codes/decoder.pth"
ENCODER_PATH = "../models_and_codes/encoderGrid.pth"
LATENT_CODE_PATH = "../models_and_codes/latent_code.pkl"
PARAM_FILE = "../config/param_encoder.yaml"
VEHICLE_VALIDATION_PATH = "../config/vehicle_validation.txt"
# VEHICLE_VALIDATION_PATH = "../config/vehicle_list_all.txt"
ANNOTATIONS_PATH = "../../results/input_images/annotations.pkl"
IMAGES_PATH = "../../results/input_images/images/"
OUTPUT_DIR = "../../results/encoder_output/evaluation"
LOGS_PATH = "../../results/logs/encoder/log.pkl"
PLOT_PATH = "../../results/plots/encoder/"

def init_xyz(resolution):
    """
    Init a grid, with 3D coordinates going from -0.5 to 0.5 in every direction.
    """
    
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz


def load_grid(list_hash, annotations, num_model_2_render, param_image, param_network):
    """ 
    Generate the 3D grid tensor from images and annotations
    """
    
    matrix_world_to_camera = annotations["matrix_world_to_camera"]

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
    """ 
    Evaluate the encoder on an input tensor
    """

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
    parser.add_argument('--num_image', type=int, help='number of images to evaluate per model -> int', default= DEFAULT_IMAGES_PER_MODEL)
    parser.add_argument('--num_model', type=int, help='number of models to evaluate, -> int or None if you want to evaluate them all', default= DEFAULT_MAX_MODEL_2_RENDER)
    parser.add_argument('--logs', type=bool, help='plots logs -> True or False', default= DEFAULT_LOGS)
    args = parser.parse_args()


    if args.render:

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

        if args.num_model is not None:
            list_hash_validation = list_hash_validation[:args.num_model]

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
        print("load grid...")
        grid = load_grid(list_hash, annotations, num_model_2_render, param["image"], param["network"])
        print("compute code from grid...")
        latent_code = get_code_from_grid(grid, param_all["latent_size"])

        
        # fill a xyz grid to give as input to the decoder 
        xyz = init_xyz(resolution)

        # load decoder
        decoder = torch.load(DECODER_PATH).cuda()
        decoder.eval()

        # list to keep in memory all the loss computed
        list_sdf = []
        list_rgb = []
        list_lab = []
        list_l2 = []
        list_cham_sdf = []
        list_cham_rgb = []
        list_cham_lab = []

        for model_hash, model_id in zip(list_hash, range(num_model)):

            print("\n")

            # decode
            sdf_target = np.empty([resolution, resolution, resolution, 4])

            for x in range(resolution):

                sdf_pred = decoder(dict_hash_2_code[model_hash].repeat(resolution * resolution, 1).cuda(),xyz[x * resolution * resolution: (x+1) * resolution * resolution]).detach()

                sdf_pred[:,0] = sdf_pred[:,0] * resolution
                sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
                sdf_pred[:,1:] = sdf_pred[:,1:] * 255

                sdf_target[x, :, :, :] = np.reshape(sdf_pred[:,:].cpu(), [resolution, resolution, 4])

            print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_target[:,:,:,0]), np.max(sdf_target[:,:,:,0])))
            if(np.min(sdf_target[:,:,:,0]) < 0 and np.max(sdf_target[:,:,:,0]) > 0):
                vertices_target, faces_target = marching_cubes(sdf_target[:,:,:,0])
                colors_v_target = exctract_colors_v(vertices_target, sdf_target)
                colors_f_target = exctract_colors_f(colors_v_target, faces_target)
                off_file = "%s/%s_target.off" %(OUTPUT_DIR, model_hash)
                write_off(off_file, vertices_target, faces_target, colors_f_target)
                print("Wrote %s_target.off" % model_hash)
            else:
                print("surface level: 0, should be comprise in between the minimum and maximum value")

            vertices_target = torch.tensor(vertices_target.copy())
            colors_v_target = torch.tensor(colors_v_target/255).unsqueeze(0).cuda()


            # decode
            sdf_validation = np.empty([resolution, resolution, resolution, 4])
            mean_code = latent_code[model_id,:,:].mean(dim=0)

            for x in range(resolution):

                sdf_pred = decoder(mean_code.repeat(resolution * resolution, 1),xyz[x * resolution * resolution: (x+1) * resolution * resolution]).detach()

                sdf_pred[:,0] = sdf_pred[:,0] * resolution
                sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
                sdf_pred[:,1:] = sdf_pred[:,1:] * 255

                sdf_validation[x, :, :, :] = np.reshape(sdf_pred[:,:].cpu(), [resolution, resolution, 4])


            print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_validation[:,:,:,0]), np.max(sdf_validation[:,:,:,0])))
            if(np.min(sdf_validation[:,:,:,0]) < 0 and np.max(sdf_validation[:,:,:,0]) > 0):
                vertices, faces = marching_cubes(sdf_validation[:,:,:,0])
                colors_v = exctract_colors_v(vertices, sdf_validation)
                colors_f = exctract_colors_f(colors_v, faces)
                off_file = '%s/%s_prediction.off' %(OUTPUT_DIR, model_hash)
                write_off(off_file, vertices, faces, colors_f)
                print('Wrote %s_prediction.off' % (model_hash))
            else:
                print("surface level: 0, should be comprise in between the minimum and maximum value")

            vertices = torch.tensor(vertices.copy())
            colors_v= torch.tensor(colors_v/255).unsqueeze(0).cuda()

            # compute the sdf from codes 
            sdf_validation = torch.tensor(sdf_validation).reshape(resolution * resolution * resolution, 4)
            sdf_target= torch.tensor(sdf_target).reshape(resolution * resolution * resolution, 4)

            # assign weight of 0 for easy samples that are well trained
            threshold_precision = 1
            weight_sdf = ~((sdf_validation[:,0] > threshold_precision).squeeze() * (sdf_target[:,0] > threshold_precision).squeeze()) \
                * ~((sdf_validation[:,0] < -threshold_precision).squeeze() * (sdf_target[:,0] < -threshold_precision).squeeze())

            # loss l1 in distance error per samples
            loss_sdf = torch.nn.L1Loss(reduction='none')(sdf_validation[:,0].squeeze(), sdf_target[:,0])
            loss_sdf = (loss_sdf * weight_sdf).mean() * weight_sdf.numel()/weight_sdf.count_nonzero()

            # loss rgb in pixel value difference per color per samples
            rgb_gt_normalized = sdf_target[:,1:]
            loss_rgb = torch.nn.L1Loss(reduction='none')(sdf_validation[:,1:], rgb_gt_normalized)
            loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).mean()/3 * weight_sdf.numel()/weight_sdf.count_nonzero()


            # lab loss
            sdf_validation[:,1:] = sdf_validation[:,1:] / 255
            sdf_validation[:,1:] = torch.tensor(color.rgb2lab(sdf_validation[:,1:]))

            sdf_target[:,1:] = sdf_target[:,1:] / 255
            sdf_target[:,1:] = torch.tensor(color.rgb2lab(sdf_target[:,1:]))

            # loss LAB in pixel value difference per color per samples
            rgb_gt_normalized = sdf_target[:,1:]
            loss_lab = torch.nn.L1Loss(reduction='none')(sdf_validation[:,1:], rgb_gt_normalized)
            loss_lab = ((loss_lab[:,0] * weight_sdf) + (loss_lab[:,1] * weight_sdf) + (loss_lab[:,2] * weight_sdf)).mean()/3 * weight_sdf.numel()/weight_sdf.count_nonzero()

            l2_error = (dict_hash_2_code[model_hash].cuda() - mean_code).norm()


            cham_sdf, cham_rgb, cham_lab = chamfer_distance_rgb(vertices, vertices_target, colors_x = colors_v, colors_y = colors_v_target)



            # fill list with all the results
            print(f"loss_sdf: {loss_sdf}")
            print(f"cham_sdf: {cham_sdf}")
            print(f"loss_rgb: {loss_rgb}")
            print(f"cham_rgb: {cham_rgb}")
            print(f"loss_lab: {loss_lab}")
            print(f"cham_lab: {cham_lab}")
            print(f"l2_error: {l2_error}")

            list_sdf.append(loss_sdf)
            list_rgb.append(loss_rgb)
            list_lab.append(loss_lab)
            list_l2.append(l2_error)

            list_cham_sdf.append(cham_sdf)
            list_cham_rgb.append(cham_rgb)
            list_cham_lab.append(cham_lab)


        # print(f"std loss_sdf: {torch.tensor(list_sdf).std()}")
        # print(f"std loss_rgb: {torch.tensor(list_rgb).std()}")
        # print(f"std loss_lab: {torch.tensor(list_lab).std()}")
        # print(f"std l2_error: {torch.tensor(list_l2).std()}")

        print("\nloss computed with GT")

        print(f"mean loss_sdf: {torch.tensor(list_sdf).mean()}")
        print(f"mean cham_sdf: {torch.tensor(list_cham_sdf).mean()}")
        print(f"mean loss_rgb: {torch.tensor(list_rgb).mean()}")
        print(f"mean cham_rgb: {torch.tensor(list_cham_rgb).mean()}")
        print(f"mean loss_lab: {torch.tensor(list_lab).mean()}")
        print(f"mean cham_lab: {torch.tensor(list_cham_lab).mean()}")
        print(f"mean l2_error: {torch.tensor(list_l2).mean()}")


        l2_dist_each_other = []
        l2_dist_mean = []

        for model in range(num_model):
            mean_code = latent_code[model,:, :].mean(dim=0)
            for im1 in range(num_model_2_render):
                l2_dist_mean.append((latent_code[model,im1, :] - mean_code).norm())
                for im2 in range(im1 + 1, num_model_2_render):
                    l2_dist_each_other.append((latent_code[model,im1, :] - latent_code[model, im2, :]).norm())

        l2_dist_each_other = torch.tensor(l2_dist_each_other)
        l2_dist_mean = torch.tensor(l2_dist_mean)


        if num_model_2_render > 1:
            print("\nl2 distance computed between prediction")
            print(f"max: {l2_dist_each_other.max()}")
            print(f"mean: {l2_dist_each_other.mean()}")
            print(f"std: {l2_dist_each_other.std()}")


            print("\nl2 distance computed with mean of prediction")
            print(f"max: {l2_dist_mean.max()}")
            print(f"mean: {l2_dist_mean.mean()}")
            print(f"std: {l2_dist_mean.std()}")


    if args.logs:

        # load parameters
        logs = pickle.load(open(LOGS_PATH, 'rb'))

        param_all = yaml.safe_load(open(PARAM_FILE))
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

        plt.figure()
        plt.title("logs loss lab validation")
        plt.semilogy(x_timestamp_validation_temp,logs["validation"]["lab"])
        plt.ylabel("error lab")
        plt.xlabel("epoch")
        plt.savefig(PLOT_PATH + "lab_val.png")


        plt.figure()
        plt.title("logs error cham sdf validation")
        plt.semilogy(x_timestamp_validation_temp,logs["validation"]["cham_sdf"])
        plt.ylabel("error cham sdf")
        plt.xlabel("epoch")
        plt.savefig(PLOT_PATH + "cham_sdf_val.png")

        plt.figure()
        plt.title("logs error cham rgb validation")
        plt.semilogy(x_timestamp_validation_temp,logs["validation"]["cham_rgb"])
        plt.ylabel("error cham rgb")
        plt.xlabel("epoch")
        plt.savefig(PLOT_PATH + "cham_rgb_val.png")

        plt.figure()
        plt.title("logs error cham lab validation")
        plt.semilogy(x_timestamp_validation_temp,logs["validation"]["cham_lab"])
        plt.ylabel("error cham lab")
        plt.xlabel("epoch")
        plt.savefig(PLOT_PATH + "cham_lab_val.png")

    print("done")


