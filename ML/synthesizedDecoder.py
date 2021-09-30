import torch
import math
from marching_cubes_rgb import *

num_synthesized_scene = 10

TESTING = False

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


# initialize random latent code 
lat_vecs_synthesized = torch.nn.Embedding(num_synthesized_scene, lat_vecs.weight.shape[1]).cuda()
torch.nn.init.normal_(
    lat_vecs_synthesized.weight.data,
    0.0,
    1.0 / math.sqrt(lat_vecs.weight.shape[1]),
)

num_scenes = len(lat_vecs_synthesized.weight)
idx = torch.arange(num_scenes).cuda()
xyz = torch.empty(num_samples_per_scene, 3,  dtype=torch.float).cuda()
for x in range(resolution):
    for y in range(resolution):
        for z in range(resolution):
            xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

decoder.eval()

for i in range(num_scenes):
    
    # free variable for memory space
    try:
        del sdf_pred
    except:
        print("sdf_pred wasn't defined")

    # decode
    sdf_result = np.empty([resolution, resolution, resolution, 4])


    for x in range(resolution):
        

        sdf_pred = decoder(lat_vecs_synthesized(idx[i].repeat(resolution * resolution)),xyz[x * resolution * resolution: (x+1) * resolution * resolution])

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
        off_file = '../../data_processing/output_synthesized/%d.off' % i
        write_off(off_file, vertices, faces, colors_f)
        print('Wrote %s.' % off_file)
    else:
        print("surface level: 0, should be comprise in between the minimum and maximum value")



# initialize random latent code 
lat_vecs_mean = torch.nn.Embedding(1, lat_vecs.weight.shape[1]).cuda()
torch.nn.init.normal_(
    lat_vecs_mean.weight.data,
    0.0,
    0.0,
)

num_scenes = len(lat_vecs_mean.weight)
idx = torch.arange(num_scenes).cuda()

# free variable for memory space
try:
    del sdf_pred
except:
    print("sdf_pred wasn't defined")

# decode
sdf_result = np.empty([resolution, resolution, resolution, 4])

for x in range(resolution):
    
    sdf_pred = decoder(lat_vecs_mean(idx[0].repeat(resolution * resolution)),xyz[x * resolution * resolution: (x+1) * resolution * resolution])

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
    off_file = '../../data_processing/output_synthesized/_mean.off'
    write_off(off_file, vertices, faces, colors_f)
    print('Wrote %s.' % off_file)
else:
    print("surface level: 0, should be comprise in between the minimum and maximum value")