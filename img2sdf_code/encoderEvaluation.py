import torch
import pickle
import json
import matplotlib.pyplot as plt

from marching_cubes_rgb import *
import IPython

DEFAULT_RENDER = True
DEFAULT_RENDER_RESOLUTION = 64
DEFAULT_MAX_MODEL_2_RENDER = 10
DEFAULT_LOGS = True


RESOLUTION_USED_IN_TRAINING = 64
DECODER_PATH = "models_and_codes/decoder.pth"
ENCODER_PATH = "models_and_codes/encoderGrid.pth"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
OUTPUT_DIR = "../../image2sdf/encoder_output/evaluation"
LOGS_PATH = "../../image2sdf/logs/encoder/log.pkl"
PLOT_PATH = "../../image2sdf/plots/encoder/"
PARAM_FILE = "config/param.json"

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
        for hash, i in zip(list_hash, range(num_model_2_render)):

            # variable to store results
            sdf_result = np.empty([resolution, resolution, resolution, 4])


            # loop because it requires too much GPU memory on my computer
            for x in range(resolution):
                latent_code = dict_hash_2_code[hash].repeat(resolution * resolution, 1).cuda()
                xyz_sub_sample = xyz[x * resolution * resolution: (x+1) * resolution * resolution]

                sdf_pred = decoder(latent_code, xyz_sub_sample).detach().cpu()
                sdf_pred[:,0] = sdf_pred[:,0] * resolution
                sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
                sdf_pred[:,1:] = sdf_pred[:,1:] * 255

                sdf_result[x, :, :, :] = np.reshape(sdf_pred[:,:], [resolution, resolution, 4])

            # print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result[:,:,:,0]), np.max(sdf_result[:,:,:,0])))
            if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
                vertices, faces = marching_cubes(sdf_result[:,:,:,0])
                colors_v = exctract_colors_v(vertices, sdf_result)
                colors_f = exctract_colors_f(colors_v, faces)
                off_file = "%s/%s.off" %(OUTPUT_DIR, hash)
                write_off(off_file, vertices, faces, colors_f)
                print("Wrote %s.off" % hash)
            else:
                print("surface level: 0, should be comprise in between the minimum and maximum value")


    if args.logs:

        # load parameters
        logs = pickle.load(open(LOGS_PATH, 'rb'))

    print("done")


