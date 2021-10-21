import pickle
import glob
import imageio
import torch
import cv2
from marching_cubes_rgb import *

import IPython

DEFAULT_RESOLUTION = 100
DEFAULT_NUM_IMAGE = 3
DEFAUT_OUTPUT_IMAGES = False
DEFAULT_TYPE = "validation"

DECODER_PATH = "models_pth/decoderSDF.pth"
ENCODER_PATH = "models_pth/encoderSDF.pth"
LATENT_VECS_PRED_PATH = "models_pth/latent_vecs_pred.pth"
MATRIX_PATH = "../../image2sdf/input_images/matrix_w2c.pkl"
ANNOTATIONS_PATH = "../../image2sdf/input_images_validation/annotations.pkl"
IMAGES_PATH = "../../image2sdf/input_images_validation/images/"

latent_size = 16

height_input_image = 300
width_input_image = 450

num_slices = 50
width_input_network = 25
height_input_network = 25
# depth_input_network = 120


def convert_w2c(matrix_world_to_camera, frame, point):

    point_4d = np.resize(point, 4)
    point_4d[3] = 1
    co_local = matrix_world_to_camera.dot(point_4d)
    z = -co_local[2]

    if z == 0.0:
            return np.array([0.5, 0.5, 0.0])
    else:
        for i in range(3):
            frame[i] =  -(frame[i] / (frame[i][2]/z))

    min_x, max_x = frame[2][0], frame[1][0]
    min_y, max_y = frame[1][1], frame[0][1]

    x = (co_local[0] - min_x) / (max_x - min_x)
    y = (co_local[1] - min_y) / (max_y - min_y)

    return np.array([x,y,z])


def load_grid(annotations, argument_num_image):

    matrix_world_to_camera = pickle.load(open(MATRIX_PATH, 'rb'))

    num_image_per_scene = len(annotations[next(iter(annotations.keys()))])
    num_scene = len(annotations.keys())
    num_image_per_scene = min(num_image_per_scene, argument_num_image)

    
    all_grid = torch.empty([num_scene, num_image_per_scene, 3, num_slices, width_input_network, height_input_network], dtype=torch.float)

    for scene, scene_id in zip(annotations.keys(), range(num_scene)):
        for image, image_id in zip(glob.glob(IMAGES_PATH + scene + '/*'), range(num_image_per_scene)):

            # Load data and get label
            image_pth = IMAGES_PATH + scene + '/' + str(image_id) + '.png'
            input_im = imageio.imread(image_pth)

            # loc_2d = self.annotations[scene_id][rand_image_id]['2d'].copy()
            loc_3d = annotations[scene][image_id]['3d'].copy()
            frame = annotations[scene][image_id]['frame'].copy()

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
            loc_slice_2d[:,:,0] = loc_slice_2d[:,:,0] * width_input_image
            loc_slice_2d[:,:,1] = loc_slice_2d[:,:,1] * height_input_image

            # grid to give as input to the network
            input_grid = np.empty([num_slices, width_input_network, height_input_network, 3])


            # fill grid by slices
            for i in range(num_slices):
                src = loc_slice_2d[i,:,:2].copy()
                dst = np.array([[0, height_input_network], [width_input_network, height_input_network], [width_input_network, 0], [0,0]])
                h, mask = cv2.findHomography(src, dst)
                slice = cv2.warpPerspective(input_im, h, (width_input_network,height_input_network))
                input_grid[i,:,:,:] = slice

            # rearange, normalize and convert to tensor
            input_grid = np.transpose(input_grid, [3,0,1,2])
            input_grid = input_grid/255 - 0.5
            input_grid = torch.tensor(input_grid, dtype = torch.float)

            all_grid[scene_id, image_id, :, :, :, :] = input_grid

    return all_grid


def get_vecs(grid):

    encoder = torch.load(ENCODER_PATH).cuda()
    encoder.eval()

    num_scene = grid.shape[0]
    num_image_per_scene = grid.shape[1]

    lat_vecs = torch.empty([num_scene, num_image_per_scene, latent_size]).cuda()
    for scene_id in range(num_scene):
        print(f"encoding scene n°: {scene_id}")
        for image_id in range(num_image_per_scene):
            lat_vecs[scene_id,image_id,:] = encoder(grid[scene_id, image_id, :, :, :, :].unsqueeze(0).cuda()).detach()

    return lat_vecs


def init_xyz(resolution):
    xyz = torch.empty(resolution * resolution * resolution, 3).cuda()

    for x in range(resolution):
        for y in range(resolution):
            for z in range(resolution):
                xyz[x * resolution * resolution + y * resolution + z, :] = torch.Tensor([x/(resolution-1)-0.5,y/(resolution-1)-0.5,z/(resolution-1)-0.5])

    return xyz

def cosine_distance(a,b):
    return a.dot(b)/(a.norm() * b.norm())

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Peform marching cubes.')
    parser.add_argument('--resolution', type=int, help='resolution', default= DEFAULT_RESOLUTION)
    parser.add_argument('--num_image', type=int, help='num max images per scene', default= DEFAULT_NUM_IMAGE)
    parser.add_argument('--output_images', type=bool, help='num max images per scene', default= DEFAUT_OUTPUT_IMAGES)
    args = parser.parse_args()



    resolution = args.resolution
    num_samples_per_scene = resolution * resolution * resolution

    annotations_file = open(ANNOTATIONS_PATH, "rb")
    annotations = pickle.load(annotations_file)

    grid = load_grid(annotations, args.num_image)
    lat_vecs = get_vecs(grid)

    num_scene = lat_vecs.shape[0]
    num_image_per_scene = lat_vecs.shape[1]

    if args.output_images:
        output_dir = "output_encoder_grid"
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

    similarity_same_model = []
    similarity_different_model = []

    for scene_id_1 in range(num_scene):
        for vec1 in range(num_image_per_scene-1):
            for scene_id_2 in range(scene_id_1, num_scene):
                if scene_id_1 == scene_id_2:
                  print(f"cosine distance for scene {scene_id_1}")
                for vec2 in range(vec1 + 1, num_image_per_scene):
                    dist = cosine_distance(lat_vecs[scene_id_1,vec1,:], lat_vecs[scene_id_2,vec2,:])
                    if scene_id_1 == scene_id_2:
                        similarity_same_model.append(dist)
                        print(f"cosine distance between vec {vec1} and {vec2}: {dist}")
                    else:
                        similarity_different_model.append(dist)


    print(f"average similarity between same models: {torch.tensor(similarity_same_model).mean()}")
    print(f"average similarity between differents models: {torch.tensor(similarity_different_model).mean()}")