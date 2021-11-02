import torch
from marching_cubes_rgb import *

import IPython


DECODER_PATH = "models_pth/decoderSDF.pth"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
OUTPUT_DIR = "../../image2sdf/output_decoder"

DEFAULT_RESOLUTION = 100
DEFAULT_MAX_MODEL_2_RENDER = 10


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
    parser.add_argument('--resolution', type=int, help='resolution', default= DEFAULT_RESOLUTION)
    parser.add_argument('--max_model', type=int, help='resolution', default= DEFAULT_MAX_MODEL_2_RENDER)
    args = parser.parse_args()


    # load decoder and codes
    decoder = torch.load(DECODER_PATH).cuda()
    dict_hash_2_code = torch.load(LATENT_CODE_PATH)

    # initialize parameters
    list_hash = dict_hash_2_code.keys()
    num_scene = len(list_hash)
    resolution = args.resolution
    num_samples_per_scene = resolution * resolution * resolution
    
    # fill a xyz grid to give as input to the decoder 
    xyz = init_xyz(resolution)

    decoder.eval()

    # loop through the models to render
    for hash, i in zip(list_hash, range(num_scene)):

        # variable to store results
        sdf_result = np.empty([resolution, resolution, resolution, 4])

        for x in range(resolution):

            sdf_pred = decoder(dict_hash_2_code[hash].repeat(resolution * resolution, 1),xyz[x * resolution * resolution: (x+1) * resolution * resolution]).detach()

            sdf_pred[:,0] = sdf_pred[:,0] * resolution
            sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
            sdf_pred[:,1:] = sdf_pred[:,1:] * 255

            sdf_result[x, :, :, :] = np.reshape(sdf_pred[:,:].detach().cpu(), [resolution, resolution, 4])

        # print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result[:,:,:,0]), np.max(sdf_result[:,:,:,0])))
        if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
            vertices, faces = marching_cubes(sdf_result[:,:,:,0])
            colors_v = exctract_colors_v(vertices, sdf_result)
            colors_f = exctract_colors_f(colors_v, faces)
            off_file = "%s/%s.off" %(OUTPUT_DIR, hash)
            write_off(off_file, vertices, faces, colors_f)
            print("Wrote %s." % hash)
        else:
            print("surface level: 0, should be comprise in between the minimum and maximum value")
