""" 
Render random code in the latent space with the decoder.
"""

import torch
import pickle

from marching_cubes_rgb import *
import IPython

DEFAULT_NUM_MODELS = 5
DEFAULT_RENDER_RESOLUTION = 64


DECODER_PATH = "../models_and_codes/decoder.pth"
LATENT_CODE_PATH = "../models_and_codes/latent_code.pkl"
OUTPUT_DIR = "../../results/decoder_output/synthesized"

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

if __name__ == '__main__':

    # read arguments
    parser = argparse.ArgumentParser(description='Render decoder results')
    parser.add_argument('--resolution', type=int, help='resolution -> int', default= DEFAULT_RENDER_RESOLUTION)
    parser.add_argument('--num_model', type=int, help='number of model to render -> int', default= DEFAULT_NUM_MODELS)
    args = parser.parse_args()

    # load decoder and codes
    decoder = torch.load(DECODER_PATH).cuda()
    dict_hash_2_code = pickle.load(open(LATENT_CODE_PATH, 'rb'))

    # initialize parameters
    num_model= args.num_model
    resolution = args.resolution

    # fill a xyz grid to give as input to the decoder 
    xyz = init_xyz(resolution)
    idx = torch.arange(num_model).cuda()

    # find std of latend codes
    list_hash = list(dict_hash_2_code.keys())
    list_code = []
    for model_hash in list_hash:
        list_code.append(np.array(dict_hash_2_code[model_hash]))

    array_code = np.array(list_code)
    std_lat_space = array_code.std()

    # initialize random latent code 
    lat_code_synthesized = torch.nn.Embedding(num_model, dict_hash_2_code[list_hash[0]].shape[0]).cuda()
    torch.nn.init.normal_(
        lat_code_synthesized.weight.data,
        0.0,
        std_lat_space,
    )

    decoder.eval()

    for i in range(num_model):
        
        # variable to store results
        sdf_result = np.empty([resolution, resolution, resolution, 4])


        # loop because it requires too much GPU memory on my computer
        for x in range(resolution):
            latent_code = lat_code_synthesized(idx[i].repeat(resolution * resolution))
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
            off_file = '%s/%d.off' % (OUTPUT_DIR, i)
            write_off(off_file, vertices, faces, colors_f)
            print('Wrote %d.off' % i)
        else:
            print("surface level: 0, should be comprise in between the minimum and maximum value")

    # initialize random latent code 
    lat_vecs_mean = torch.nn.Embedding(1, dict_hash_2_code[list_hash[0]].shape[0]).cuda()
    torch.nn.init.normal_(
        lat_vecs_mean.weight.data,
        0.0,
        0.0,
    )

    # variable to store results
    sdf_result_mean = np.empty([resolution, resolution, resolution, 4])

    for x in range(resolution):
        sdf_pred = decoder(lat_vecs_mean(idx[0].repeat(resolution * resolution)),xyz[x * resolution * resolution: (x+1) * resolution * resolution]).detach()

        sdf_pred[:,0] = sdf_pred[:,0] * resolution
        sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
        sdf_pred[:,1:] = sdf_pred[:,1:] * 255

        sdf_result_mean[x, :, :, :] = np.reshape(sdf_pred[:,:].cpu(), [resolution, resolution, 4])

    # print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result_mean[:,:,:,0]), np.max(sdf_result_mean[:,:,:,0])))
    if(np.min(sdf_result_mean[:,:,:,0]) < 0 and np.max(sdf_result_mean[:,:,:,0]) > 0):
        vertices, faces = marching_cubes(sdf_result_mean[:,:,:,0])
        colors_v = exctract_colors_v(vertices, sdf_result_mean)
        colors_f = exctract_colors_f(colors_v, faces)
        off_file = '%s/_mean.off' % OUTPUT_DIR
        write_off(off_file, vertices, faces, colors_f)
        print('Wrote _mean.off')
    else:
        print("surface level: 0, should be comprise in between the minimum and maximum value")
