import warnings
import torch
import pickle
# import json
import yaml
from skimage import color
import matplotlib.pyplot as plt

from marching_cubes_rgb import *
from utils import chamfer_distance_rgb
import IPython

DEFAULT_RENDER = True
# DEFAULT_RENDER = False
DEFAULT_RENDER_RESOLUTION = 64
DEFAULT_MAX_MODEL_2_RENDER = 10
DEFAULT_LOGS = True


warnings.filterwarnings("ignore")


DECODER_PATH = "models_and_codes/decoder.pth"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
OUTPUT_DIR = "../../image2sdf/decoder_output/evaluation"
SDF_DIR = "../../image2sdf/sdf/"
LOGS_PATH = "../../image2sdf/logs/decoder/log.pkl"
PLOT_PATH = "../../image2sdf/plots/decoder/"
# PARAM_FILE = "config/param.json"
PARAM_FILE = "config/param_decoder.yaml"

def init_xyz(resolution):
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz


if __name__ == '__main__':

    # read arguments
    parser = argparse.ArgumentParser(description='Render decoder results')
    parser.add_argument('--render', type=bool, help='render model -> True or False', default= DEFAULT_RENDER)
    parser.add_argument('--resolution', type=int, help='resolution -> int', default= DEFAULT_RENDER_RESOLUTION)
    parser.add_argument('--max_model', type=int, help='max number of model to render -> int', default= DEFAULT_MAX_MODEL_2_RENDER)
    parser.add_argument('--logs', type=bool, help='plots logs -> True or False', default= DEFAULT_LOGS)
    args = parser.parse_args()


    if args.render:

        list_cham_sdf = []
        list_cham_rgb = []
        list_cham_lab = []

        # load decoder and codes
        decoder = torch.load(DECODER_PATH).cuda()
        dict_hash_2_code = pickle.load(open(LATENT_CODE_PATH, 'rb'))

        # initialize parameters
        list_hash = dict_hash_2_code.keys()
        num_model_2_render= min(len(list_hash), args.max_model)
        resolution = args.resolution
        
        # fill a xyz grid to give as input to the decoder 
        xyz = init_xyz(resolution)

        decoder.eval()

        # loop through the models to render
        for model_hash, i in zip(list_hash, range(num_model_2_render)):

            # variable to store results
            sdf_result = np.empty([resolution, resolution, resolution, 4])

            # loop because it requires too much GPU memory on my computer
            for x in range(resolution):
                latent_code = dict_hash_2_code[model_hash].repeat(resolution * resolution, 1).cuda()
                xyz_sub_sample = xyz[x * resolution * resolution: (x+1) * resolution * resolution]

                sdf_pred = decoder(latent_code, xyz_sub_sample).detach().cpu()
                sdf_pred[:,0] = sdf_pred[:,0] * resolution
                sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)

    ######################################## used for LAB testing ########################################
                # sdf_pred[:,1] = (sdf_pred[:,1]) * 100
                # sdf_pred[:,2:] = (sdf_pred[:,2:] - 0.5) * 200
                # sdf_pred[:,1:] = torch.tensor(color.lab2rgb(sdf_pred[:,1:]))
    ######################################## used for LAB testing ########################################

                sdf_pred[:,1:] = sdf_pred[:,1:] * 255

                sdf_result[x, :, :, :] = np.reshape(sdf_pred[:,:], [resolution, resolution, 4])

            # print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result[:,:,:,0]), np.max(sdf_result[:,:,:,0])))
            if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
                vertices_pred, faces_pred = marching_cubes(sdf_result[:,:,:,0])
                colors_v_pred = exctract_colors_v(vertices_pred, sdf_result)
                colors_f_pred = exctract_colors_f(colors_v_pred, faces_pred)
                off_file = "%s/%s_rgb.off" %(OUTPUT_DIR, model_hash)
                # write_off(off_file, vertices_pred, faces_pred, colors_f_pred)
                # print("Wrote %s_rgb.off" % model_hash)
            else:
                print("surface level: 0, should be comprise in between the minimum and maximum value")

            h5f = h5py.File(SDF_DIR + model_hash + '.h5', 'r')
            sdf_gt = h5f["tensor"][()]

            # print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result[:,:,:,0]), np.max(sdf_result[:,:,:,0])))
            if(np.min(sdf_gt[:,:,:,0]) < 0 and np.max(sdf_gt[:,:,:,0]) > 0):
                vertices_gt, faces_gt = marching_cubes(sdf_gt[:,:,:,0])
                colors_v_gt = exctract_colors_v(vertices_gt, sdf_gt)
                colors_f_gt = exctract_colors_f(colors_v_gt, faces_gt)
                off_file = "%s/%s_gt.off" %(OUTPUT_DIR, model_hash)
                # write_off(off_file, vertices_gt, faces_gt, colors_f_gt)
                # print("Wrote %s_gt.off" % model_hash)
            else:
                print("surface level: 0, should be comprise in between the minimum and maximum value")

            # compute the sdf from codes 
            sdf_validation = torch.tensor(sdf_result).reshape(resolution * resolution * resolution, 4)
            sdf_target= torch.tensor(sdf_gt).reshape(resolution * resolution * resolution, 4)


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

            # lab loss
            sdf_validation[:,1:] = sdf_validation[:,1:] / 255
            sdf_validation[:,1:] = torch.tensor(color.rgb2lab(sdf_validation[:,1:]))

            sdf_target[:,1:] = sdf_target[:,1:] / 255
            sdf_target[:,1:] = torch.tensor(color.rgb2lab(sdf_target[:,1:]))

            # loss rgb in pixel value difference per color per samples
            rgb_gt_normalized = sdf_target[:,1:]
            loss_lab = torch.nn.L1Loss(reduction='none')(sdf_validation[:,1:], rgb_gt_normalized)
            loss_lab = ((loss_lab[:,0] * weight_sdf) + (loss_lab[:,1] * weight_sdf) + (loss_lab[:,2] * weight_sdf)).mean()/3 * weight_sdf.numel()/weight_sdf.count_nonzero()

            print(f"loss_lab: {loss_lab}")


            cham_sdf, cham_rgb, cham_lab = chamfer_distance_rgb(vertices_pred, vertices_gt, colors_x = colors_v_pred, colors_y = colors_v_gt)

            print(f"cham sdf: {cham_sdf}")
            print(f"cham rgb: {cham_rgb}")
            print(f"cham lab: {cham_lab}")

            list_cham_sdf.append(cham_sdf)
            list_cham_rgb.append(cham_rgb)
            list_cham_lab.append(cham_lab)

    
        print(f"mean cham sdf: {torch.tensor(list_cham_sdf).mean()}")
        print(f"mean cham rgb: {torch.tensor(list_cham_rgb).mean()}")
        print(f"mean cham lab: {torch.tensor(list_cham_lab).mean()}")


    if args.logs:

        # load parameters
        logs = pickle.load(open(LOGS_PATH, 'rb'))
        dict_hash_2_code = pickle.load(open(LATENT_CODE_PATH, 'rb'))

        param_all = yaml.safe_load(open(PARAM_FILE))
        param = param_all["decoder"]

        list_hash = list(dict_hash_2_code.keys())
        num_model = len(list_hash)

        list_norm = []
        for model_hash in list_hash:
            list_norm.append(dict_hash_2_code[model_hash].norm())

        num_batch_per_epoch = param_all["resolution_used_for_training"] **3 * num_model/ param["dataLoader"]["batch_size"] / param["num_batch_between_print"]
        x_timestamp = np.arange(len(logs["sdf"])) / num_batch_per_epoch

        avrg_time = 10
        avrg_sdf =[]
        avrg_rgb =[]

        for i in range(len(x_timestamp)):
            avrg_sdf.append(torch.tensor(logs["sdf"][i-avrg_time : i + avrg_time]).mean())
            avrg_rgb.append(torch.tensor(logs["rgb"][i-avrg_time : i + avrg_time]).mean())


        # let's plots :)
        # sdf
        plt.figure()
        plt.title("logs loss sdf")
        plt.semilogy(x_timestamp, logs["sdf"], 'b', label = 'all data')
        plt.semilogy(x_timestamp, avrg_sdf, 'r', label = 'avrg')
        plt.ylabel("loss sdf")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(PLOT_PATH + "sdf.png")

        # rgb
        plt.figure()
        plt.title("logs loss rgb")
        plt.semilogy(x_timestamp,logs["rgb"], 'b', label = 'all data')
        plt.semilogy(x_timestamp,avrg_rgb, 'r', label = 'avrg')
        plt.ylabel("loss rgb")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(PLOT_PATH + "rgb.png")

        # norms
        plt.figure()
        plt.title("norm of vecors")
        plt.plot(list_norm)
        plt.ylabel("norm")
        plt.savefig(PLOT_PATH + "norms.png")

        # dist same model
        plt.figure()
        plt.title("logs dist models")
        plt.plot(x_timestamp,logs["l2_dup"], 'b', label = 'duplicate')
        plt.plot(x_timestamp,logs["l2_rand"], 'r', label = 'differents')
        plt.ylabel("l2 dist")
        plt.xlabel("epoch")
        plt.legend()
        plt.savefig(PLOT_PATH + "dist_duplicate_models.png")

    print("done")


