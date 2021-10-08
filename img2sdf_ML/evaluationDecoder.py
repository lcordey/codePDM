import torch
from traitlets.traitlets import default
from marching_cubes_rgb import *

import IPython


MODEL_PATH = "models_pth/decoderSDF.pth"
LATENT_VECS_TARGET_PATH = "models_pth/latent_vecs_target.pth"
LATENT_VECS_PRED_PATH = "models_pth/latent_vecs_pred.pth"


DEFAULT_RESOLUTION = 50
DEFAULT_NUM_IMAGE = 3
DEFAULT_TYPE = "pred"

def load_vecs(type):
    if type == "target":
        lat_vecs = torch.load(LATENT_VECS_TARGET_PATH).unsqueeze(1).cuda()
        output_dir = "output_decoder"
    elif type == "pred":
        lat_vecs = torch.load(LATENT_VECS_PRED_PATH).cuda()
        output_dir = "output_encoder"
    
    return lat_vecs, output_dir

def init_xyz(resolution):
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Peform marching cubes.')
    parser.add_argument('--type', type=str, help='"target"or "pred"', default= DEFAULT_TYPE)
    parser.add_argument('--resolution', type=int, help='resolution', default= DEFAULT_RESOLUTION)
    parser.add_argument('--num_image', type=int, help='num max images per scene', default= DEFAULT_NUM_IMAGE)
    args = parser.parse_args()

    assert(args.type == "pred" or args.type == "target"), "please precise which latent vectors you want to evaluate -> pred or target"


    resolution = args.resolution
    num_samples_per_scene = resolution * resolution * resolution

    decoder = torch.load(MODEL_PATH).cuda()

    lat_vecs, output_dir = load_vecs(args.type)

    num_scene = lat_vecs.shape[0]
    num_image = min(lat_vecs.shape[1], args.num_image)
    idx = torch.arange(num_scene).cuda()
    xyz = init_xyz(resolution)

    decoder.eval()

    for i in range(num_scene):
        for j in range(num_image):
        
            # # free variable for memory space --> no need since ".detach" was added after decoder evaluation
            # try:
            #     del sdf_pred
            # except:
            #     print("sdf_pred wasn't defined")

            # decode
            sdf_result = np.empty([resolution, resolution, resolution, 4])

            for x in range(resolution):

                sdf_pred = decoder(lat_vecs[i,j,:].repeat(resolution * resolution, 1),xyz[x * resolution * resolution: (x+1) * resolution * resolution]).detach()

                sdf_pred[:,0] = sdf_pred[:,0] * resolution
                sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
                sdf_pred[:,1:] = sdf_pred[:,1:] * 255

                sdf_result[x, :, :, :] = np.reshape(sdf_pred[:,:].detach().cpu(), [resolution, resolution, 4])


            print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result[:,:,:,0]), np.max(sdf_result[:,:,:,0])))
            if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
                vertices, faces = marching_cubes(sdf_result[:,:,:,0])
                colors_v = exctract_colors_v(vertices, sdf_result)
                colors_f = exctract_colors_f(colors_v, faces)
                off_file = '../../image2sdf/%s/%d_%d.off' %(output_dir, i, j)
                write_off(off_file, vertices, faces, colors_f)
                print('Wrote %s.' % off_file)
            else:
                print("surface level: 0, should be comprise in between the minimum and maximum value")
