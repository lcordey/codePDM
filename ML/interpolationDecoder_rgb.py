import torch

from marching_cubes_rgb import *

TESTING = True

id_car_A = 1
id_car_B = 8
num_interp = 9

resolution = 100
num_samples_per_scene = resolution * resolution * resolution

MODEL_PATH = "models_pth/decoderSDF.pth"
LATENT_VECS_PATH = "models_pth/latent_vecs.pth"

MODEL_PATH_TEST = "models_pth/decoderSDF_TEST.pth"
LATENT_VECS_PATH_TEST = "models_pth/latent_vecs_TEST.pth"


if (TESTING == True):
    decoder = torch.load(MODEL_PATH_TEST).cuda()
    lat_vecs = torch.load(LATENT_VECS_PATH_TEST).cuda()
else:
    decoder = torch.load(MODEL_PATH).cuda()
    lat_vecs = torch.load(LATENT_VECS_PATH).cuda()

num_scenes = len(lat_vecs.weight)
idx = torch.arange(num_scenes).cuda()
xyz = torch.empty(num_samples_per_scene, 3,  dtype=torch.float).cuda()

for x in range(resolution):
    for y in range(resolution):
        for z in range(resolution):
            xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])



decoder.eval()
for i in range(num_interp):
    interp_late_vecs = (i/(num_interp-1) * lat_vecs(idx[id_car_A]) + (1 - i/(num_interp-1)) * lat_vecs(idx[id_car_B])).repeat(resolution * resolution,1)
    
    # free variable for memory space
    try:
        del sdf_pred
    except:
        print("sdf_pred wasn't defined")

    # decode

    sdf_result = np.empty([resolution, resolution, resolution, 4])

    for x in range(resolution):

        sdf_pred = decoder(interp_late_vecs,xyz[x * resolution * resolution: (x+1) * resolution * resolution])
        sdf_pred[:,0] = sdf_pred[:,0] * resolution
        sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
        sdf_pred[:,1:] = sdf_pred[:,1:] * 255

        # for y in range(resolution):
        #     for z in range(resolution):
        #         sdf_result[x,y,z,:] = sdf_pred[y * resolution + z,:].detach().cpu()


        sdf_result[x, :, :, :] = np.reshape(sdf_pred[:,:].detach().cpu(), [resolution, resolution, 4])

    print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result[:,:,:,0]), np.max(sdf_result[:,:,:,0])))
    if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
        vertices, faces = marching_cubes(sdf_result[:,:,:,0])
        colors_v = exctract_colors_v(vertices, sdf_result)
        colors_f = exctract_colors_f(colors_v, faces)
        off_file = '../../data_processing/output_interpolation/%d.off' % i
        write_off(off_file, vertices, faces, colors_f)
        print('Wrote %s.' % off_file)
    else:
        print("surface level: 0, should be comprise in between the minimum and maximum value")