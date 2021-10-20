import torch
from traitlets.traitlets import default
from marching_cubes_rgb import *

import IPython


MODEL_PATH = "models_pth/decoderSDF.pth"
LATENT_VECS_TARGET_PATH = "models_pth/latent_vecs_target.pth"

DEFAULT_RESOLUTION = 100


def init_xyz(resolution):
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Peform marching cubes.')
    parser.add_argument('--resolution', type=int, help='resolution', default= DEFAULT_RESOLUTION)
    args = parser.parse_args()

    resolution = args.resolution
    num_samples_per_scene = resolution * resolution * resolution

    decoder = torch.load(MODEL_PATH).cuda()

    lat_vecs = torch.load(LATENT_VECS_TARGET_PATH).cuda()
    output_dir = "output_decoder"

    num_scene = lat_vecs.shape[0]
    idx = torch.arange(num_scene).cuda()
    xyz = init_xyz(resolution)

    decoder.eval()

    for i in range(num_scene):

        # decode
        sdf_result = np.empty([resolution, resolution, resolution, 4])

        for x in range(resolution):

            sdf_pred = decoder(lat_vecs[i,:].repeat(resolution * resolution, 1),xyz[x * resolution * resolution: (x+1) * resolution * resolution]).detach()

            sdf_pred[:,0] = sdf_pred[:,0] * resolution
            sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
            sdf_pred[:,1:] = sdf_pred[:,1:] * 255

            sdf_result[x, :, :, :] = np.reshape(sdf_pred[:,:].detach().cpu(), [resolution, resolution, 4])


        print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result[:,:,:,0]), np.max(sdf_result[:,:,:,0])))
        if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
            vertices, faces = marching_cubes(sdf_result[:,:,:,0])
            colors_v = exctract_colors_v(vertices, sdf_result)
            colors_f = exctract_colors_f(colors_v, faces)
            off_file = '../../image2sdf/%s/%d.off' %(output_dir, i)
            write_off(off_file, vertices, faces, colors_f)
            print('Wrote %s.' % off_file)
        else:
            print("surface level: 0, should be comprise in between the minimum and maximum value")
