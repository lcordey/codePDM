import pickle
import glob
import imageio
import torch
from marching_cubes_rgb import *

import IPython

DEFAULT_RESOLUTION = 50
DEFAULT_NUM_IMAGE = 3
DEFAULT_TYPE = "validation"

DECODER_PATH = "models_pth/decoderSDF_VAE.pth"
ENCODER_PATH = "models_pth/encoderSDF_VAE.pth"
LATENT_VECS_PRED_PATH = "models_pth/latent_vecs_pred_vae.pth"
ANNOTATIONS_PATH = "../../image2sdf/input_images_validation/annotations.pkl"
IMAGES_PATH = "../../image2sdf/input_images_validation/images/"
# ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
# IMAGES_PATH = "../../image2sdf/input_images/images/"



def load_from_validation_data(annotations, argument_num_image):
    num_image_per_scene = len(annotations[next(iter(annotations.keys()))])
    num_scene = len(annotations.keys())
    num_image_per_scene = min(num_image_per_scene, argument_num_image)

    input_images = None
    input_locations = np.empty([num_scene, num_image_per_scene, 20])

    for scene, scene_id in zip(annotations.keys(), range(num_scene)):
        for image, image_id in zip(glob.glob(IMAGES_PATH + scene + '/*'), range(num_image_per_scene)):
        # for image_id in range(min(num_scene, num_image)):
            # image = glob.glob(IMAGES_PATH + scene + '/*')[image_id]

            # save image
            im = imageio.imread(image)

            if input_images is None:
                height = im.shape[0]
                width = im.shape[1]

                input_images = np.empty([num_scene, num_image_per_scene, im.shape[2], im.shape[0], im.shape[1]])

            input_images[scene_id, image_id, :,:,:] = np.transpose(im,(2,0,1))

            # save locations
            for loc, loc_id in zip(annotations[scene][image_id].keys(), range(len(annotations[scene][image_id].keys()))):
                if loc[-1] == 'x' or loc[-5:] == 'width':
                    input_locations[scene_id, image_id, loc_id] = annotations[scene][image_id][loc]/width
                else:
                    input_locations[scene_id, image_id, loc_id] = annotations[scene][image_id][loc]/height

    input_locations = input_locations - 0.5
    input_images = input_images/255 - 0.5

    input_locations = torch.tensor(input_locations, dtype = torch.float).cuda()
    input_images = torch.tensor(input_images, dtype = torch.float).cuda()
    
    encoder = torch.load(ENCODER_PATH).cuda()
    encoder.eval()

    latent_size = encoder(torch.empty([1,3,input_images.shape[3], input_images.shape[4]]).cuda(), torch.empty([1, input_locations.shape[2]]).cuda()).shape[1]

    lat_vecs = torch.empty([num_scene, num_image_per_scene, latent_size]).cuda()
    for scene_id in range(num_scene):
        for image_id in range(num_image_per_scene):
            lat_vecs[scene_id,image_id,:] = encoder(input_images[scene_id, image_id, :, :, :].unsqueeze(0), input_locations[scene_id, image_id, :].unsqueeze(0)).detach()

    return lat_vecs

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

    assert(args.type == "training" or args.type == "validation"), "please precise which latent vectors you want to evaluate -> pred or target"


    resolution = args.resolution
    num_samples_per_scene = resolution * resolution * resolution

    annotations_file = open(ANNOTATIONS_PATH, "rb")
    annotations = pickle.load(annotations_file)

    if args.type == "training":
        lat_vecs = torch.load(LATENT_VECS_PRED_PATH).cuda()
        output_dir = "VAE_training_prediction_vae"
    elif args.type == "validation":
        lat_vecs = load_from_validation_data(annotations, args.num_image)
        output_dir = "VAE_validation_prediction"

    num_scene = lat_vecs.shape[0]
    num_image_per_scene = lat_vecs.shape[1]
    idx = torch.arange(num_scene).cuda()
    xyz = init_xyz(resolution)


    decoder = torch.load(DECODER_PATH).cuda()
    decoder.eval()

    for scene, scene_id in zip(annotations.keys(), range(num_scene)):
        for j in range(num_image_per_scene):

            # decode
            sdf_result = np.empty([resolution, resolution, resolution, 4])

            for x in range(resolution):

                sdf_pred = decoder(lat_vecs[scene_id,j,:].repeat(resolution * resolution, 1),xyz[x * resolution * resolution: (x+1) * resolution * resolution]).detach()

                sdf_pred[:,0] = sdf_pred[:,0] * resolution
                sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
                sdf_pred[:,1:] = sdf_pred[:,1:] * 255

                sdf_result[x, :, :, :] = np.reshape(sdf_pred[:,:].detach().cpu(), [resolution, resolution, 4])


            print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result[:,:,:,0]), np.max(sdf_result[:,:,:,0])))
            if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
                vertices, faces = marching_cubes(sdf_result[:,:,:,0])
                colors_v = exctract_colors_v(vertices, sdf_result)
                colors_f = exctract_colors_f(colors_v, faces)
                off_file = '../../image2sdf/%s/%s_%d.off' %(output_dir, scene, j)
                write_off(off_file, vertices, faces, colors_f)
                print('Wrote %s.' % off_file)
            else:
                print("surface level: 0, should be comprise in between the minimum and maximum value")
