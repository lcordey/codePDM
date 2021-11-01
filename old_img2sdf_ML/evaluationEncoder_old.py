from operator import le
import pickle
import glob
import imageio
import torch
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from marching_cubes_rgb import *

import IPython

DEFAULT_RESOLUTION = 100
DEFAULT_NUM_IMAGE = 1
DEFAUT_OUTPUT_IMAGES = True
DEFAULT_TYPE = "grid"
# DEFAULT_TYPE = "face"

DECODER_PATH = "models_pth/decoderSDF.pth"
ENCODER_GRID_PATH = "models_pth/encoderGrid.pth"
ENCODER_FACE_PATH = "models_pth/encoderFace.pth"
LATENT_VECS_PRED_PATH = "models_pth/latent_vecs_pred.pth"
MATRIX_PATH = "../../image2sdf/input_images/matrix_w2c.pkl"
# ANNOTATIONS_PATH = "../../image2sdf/input_images_validation/annotations.pkl"
# IMAGES_PATH = "../../image2sdf/input_images_validation/images/"

ANNOTATIONS_PATH = "../../image2sdf/input_images/annotations.pkl"
IMAGES_PATH = "../../image2sdf/input_images/images/"
LATENT_VECS_TARGET_PATH = "models_pth/latent_vecs_target.pth"

NUM_SCENE_VALIDATION = 5

latent_size = 16

height_input_image = 300
width_input_image = 300

num_slices = 48
width_input_network_grid = 24
height_input_network_grid = 24

width_input_network_face = 64
height_input_network_face = 64
depth_input_network = 128

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx

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
    # num_scene = len(annotations.keys())
    num_scene = NUM_SCENE_VALIDATION
    num_image_per_scene = min(num_image_per_scene, argument_num_image)

    width_input_network = width_input_network_grid
    height_input_network = height_input_network_grid

    
    all_grid = torch.empty([num_scene, num_image_per_scene, 3, num_slices, width_input_network, height_input_network], dtype=torch.float)

    list_id = list(annotations.keys())

    # for scene, scene_id in zip(annotations.keys(), range(num_scene)):
    for scene, scene_id in zip(list_id[-num_scene:], range(num_scene)):
        for image, image_id in zip(glob.glob(IMAGES_PATH + scene + '/*'), range(num_image_per_scene)):

            # Load data and get label
            image_pth = IMAGES_PATH + scene + '/' + str(image_id) + '.png'
            input_im = imageio.imread(image_pth)

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



def load_face(annotations, argument_num_image):

    num_image_per_scene = len(annotations[next(iter(annotations.keys()))])
    # num_scene = len(annotations.keys())
    num_scene = 5
    num_image_per_scene = min(num_image_per_scene, argument_num_image)

    width_input_network = width_input_network_face
    height_input_network = height_input_network_face

    
    all_front = torch.empty([num_scene, num_image_per_scene, 3, width_input_network, height_input_network], dtype=torch.float)
    all_left = torch.empty([num_scene, num_image_per_scene, 3, width_input_network, depth_input_network], dtype=torch.float)
    all_back = torch.empty([num_scene, num_image_per_scene, 3, width_input_network, height_input_network], dtype=torch.float)
    all_right = torch.empty([num_scene, num_image_per_scene, 3, width_input_network, depth_input_network], dtype=torch.float)
    all_top = torch.empty([num_scene, num_image_per_scene, 3, depth_input_network, width_input_network], dtype=torch.float)

    for scene, scene_id in zip(annotations.keys(), range(num_scene)):
        for image, image_id in zip(glob.glob(IMAGES_PATH + scene + '/*'), range(num_image_per_scene)):

            # Load data and get label
            image_pth = IMAGES_PATH + scene + '/' + str(image_id) + '.png'
            input_im = imageio.imread(image_pth)

            loc_2d = annotations[scene][image_id]['2d'].copy()

            ###### y coordinate is inverted + rescaling #####
            loc_2d[:,1] = 1 - loc_2d[:,1]
            loc_2d[:,0] = loc_2d[:,0] * width_input_image
            loc_2d[:,1] = loc_2d[:,1] * height_input_image

            
            # front
            src = np.array([loc_2d[0,:2],loc_2d[1,:2],loc_2d[2,:2],loc_2d[3,:2]]).copy()
            dst = np.array([[0,height_input_network],[width_input_network,height_input_network],[width_input_network,0],[0,0]])
            h, mask = cv2.findHomography(src, dst)
            front = cv2.warpPerspective(input_im, h, (width_input_network,height_input_network))

            # left
            src = np.array([loc_2d[1,:2],loc_2d[5,:2],loc_2d[6,:2],loc_2d[2,:2]]).copy()
            dst = np.array([[0,height_input_network],[depth_input_network,height_input_network],[depth_input_network,0],[0,0]])
            h, mask = cv2.findHomography(src, dst)
            left = cv2.warpPerspective(input_im, h, (depth_input_network,height_input_network))

            # back
            src = np.array([loc_2d[5,:2],loc_2d[4,:2],loc_2d[7,:2],loc_2d[6,:2]]).copy()
            dst = np.array([[0,height_input_network],[width_input_network,height_input_network],[width_input_network,0],[0,0]])
            h, mask = cv2.findHomography(src, dst)
            back = cv2.warpPerspective(input_im, h, (width_input_network,height_input_network))

            # right
            src = np.array([loc_2d[4,:2],loc_2d[0,:2],loc_2d[3,:2],loc_2d[7,:2]]).copy()
            dst = np.array([[0,height_input_network],[depth_input_network,height_input_network],[depth_input_network,0],[0,0]])
            h, mask = cv2.findHomography(src, dst)
            right = cv2.warpPerspective(input_im, h, (depth_input_network,height_input_network))

            # top
            src = np.array([loc_2d[3,:2],loc_2d[2,:2],loc_2d[6,:2],loc_2d[7,:2]]).copy()
            dst = np.array([[0,depth_input_network],[width_input_network,depth_input_network],[width_input_network,0],[0,0]])
            h, mask = cv2.findHomography(src, dst)
            top = cv2.warpPerspective(input_im, h, (width_input_network,depth_input_network))

            # rearange, normalize and convert to tensor
            front = np.transpose(front, [2,0,1])
            front = front/255 - 0.5
            front = torch.tensor(front, dtype = torch.float)

            left = np.transpose(left, [2,0,1])
            left = left/255 - 0.5
            left = torch.tensor(left, dtype = torch.float)

            back = np.transpose(back, [2,0,1])
            back = back/255 - 0.5
            back = torch.tensor(back, dtype = torch.float)

            right = np.transpose(right, [2,0,1])
            right = right/255 - 0.5
            right = torch.tensor(right, dtype = torch.float)

            top = np.transpose(top, [2,0,1])
            top = top/255 - 0.5
            top = torch.tensor(top, dtype = torch.float)


            all_front[scene_id, image_id, :, :, :] = front
            all_left[scene_id, image_id, :, :, :] = left
            all_back[scene_id, image_id, :, :, :] = back
            all_right[scene_id, image_id, :, :, :] = right
            all_top[scene_id, image_id, :, :, :] = top

    return all_front, all_left, all_back, all_right, all_top

def get_vecs_grid(grid):

    encoder = torch.load(ENCODER_GRID_PATH).cuda()
    encoder.eval()

    num_scene = grid.shape[0]
    num_image_per_scene = grid.shape[1]

    lat_vecs = torch.empty([num_scene, num_image_per_scene, latent_size]).cuda()
    for scene_id in range(num_scene):
        # print(f"encoding scene n°: {scene_id}")
        for image_id in range(num_image_per_scene):
            lat_vecs[scene_id,image_id,:] = encoder(grid[scene_id, image_id, :, :, :, :].unsqueeze(0).cuda()).detach()

    # print(lat_vecs)

    return lat_vecs

def get_vecs_face(front, left, back, right, top):

    encoder = torch.load(ENCODER_FACE_PATH).cuda()
    encoder.eval()

    num_scene = front.shape[0]
    num_image_per_scene = front.shape[1]

    lat_vecs = torch.empty([num_scene, num_image_per_scene, latent_size]).cuda()
    for scene_id in range(num_scene):
        # print(f"encoding scene n°: {scene_id}")
        for image_id in range(num_image_per_scene):
            input_front = front[scene_id, image_id, :, :, :].unsqueeze(0).cuda()
            input_left = left[scene_id, image_id, :, :, :].unsqueeze(0).cuda()
            input_back = back[scene_id, image_id, :, :, :].unsqueeze(0).cuda()
            input_right = right[scene_id, image_id, :, :, :].unsqueeze(0).cuda()
            input_top = top[scene_id, image_id, :, :, :].unsqueeze(0).cuda()

            lat_vecs[scene_id,image_id,:] = encoder(input_front, input_left, input_back, input_right, input_top).detach()

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
    parser.add_argument('--type', type=str, help='grid or face', default= DEFAULT_TYPE)
    parser.add_argument('--output_images', type=bool, help='num max images per scene', default= DEFAUT_OUTPUT_IMAGES)
    parser.add_argument('--resolution', type=int, help='resolution', default= DEFAULT_RESOLUTION)
    parser.add_argument('--num_image', type=int, help='num max images per scene', default= DEFAULT_NUM_IMAGE)
    args = parser.parse_args()

    assert(args.type == 'grid' or args.type == 'face'), "please give type: either grid or face"

    resolution = args.resolution
    num_samples_per_scene = resolution * resolution * resolution

    annotations_file = open(ANNOTATIONS_PATH, "rb")
    annotations = pickle.load(annotations_file)

    if args.type == 'grid':
        grid = load_grid(annotations, args.num_image)
        lat_vecs = get_vecs_grid(grid)
        output_dir = "output_encoder_grid"
    else: 
        front, left, back, right, top = load_face(annotations, args.num_image)
        lat_vecs = get_vecs_face(front, left, back, right, top)
        output_dir = "output_encoder_face"

    target_vecs = torch.load(LATENT_VECS_TARGET_PATH)

    # num_scene = lat_vecs.shape[0]
    num_scene = NUM_SCENE_VALIDATION
    target_vecs = target_vecs[-num_scene:]

    num_image_per_scene = lat_vecs.shape[1]

    if args.output_images:

        # idx = torch.arange(num_scene).cuda()
        xyz = init_xyz(resolution)

        decoder = torch.load(DECODER_PATH).cuda()
        decoder.eval()

        list_id = list(annotations.keys())

        # for scene, scene_id in zip(annotations.keys(), range(num_scene)):
        for scene, scene_id in zip(list_id[-num_scene:], range(num_scene)):
            print(f"scene: {scene}")
            for j in range(num_image_per_scene):
            
                # decode
                sdf_result = np.empty([resolution, resolution, resolution, 4])
                sdf_result_for_error_computation = np.empty([resolution, resolution, resolution, 4])

                for x in range(resolution):

                    sdf_pred = decoder(lat_vecs[scene_id,j,:].repeat(resolution * resolution, 1),xyz[x * resolution * resolution: (x+1) * resolution * resolution]).detach()
                    sdf_result_for_error_computation[x, :, :, :] = np.reshape(sdf_pred[:,:].cpu(), [resolution, resolution, 4])

                    sdf_pred[:,0] = sdf_pred[:,0] * resolution
                    sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
                    sdf_pred[:,1:] = sdf_pred[:,1:] * 255

                    sdf_result[x, :, :, :] = np.reshape(sdf_pred[:,:].cpu(), [resolution, resolution, 4])


                # print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result[:,:,:,0]), np.max(sdf_result[:,:,:,0])))
                if(np.min(sdf_result[:,:,:,0]) < 0 and np.max(sdf_result[:,:,:,0]) > 0):
                    vertices, faces = marching_cubes(sdf_result[:,:,:,0])
                    colors_v = exctract_colors_v(vertices, sdf_result)
                    colors_f = exctract_colors_f(colors_v, faces)
                    off_file = '../../image2sdf/%s/%s_%d.off' %(output_dir, scene, j)
                    write_off(off_file, vertices, faces, colors_f)
                    # print('Wrote %s.' % off_file)
                else:
                    print("surface level: 0, should be comprise in between the minimum and maximum value")

        
            # decode
            sdf_result_target = np.empty([resolution, resolution, resolution, 4])
            sdf_result_target_for_error_computation = np.empty([resolution, resolution, resolution, 4])

            for x in range(resolution):

                sdf_pred = decoder(target_vecs[scene_id,:].repeat(resolution * resolution, 1),xyz[x * resolution * resolution: (x+1) * resolution * resolution]).detach()
                sdf_result_target_for_error_computation[x, :, :, :] = np.reshape(sdf_pred[:,:].cpu(), [resolution, resolution, 4])

                sdf_pred[:,0] = sdf_pred[:,0] * resolution
                sdf_pred[:,1:] = torch.clamp(sdf_pred[:,1:], 0, 1)
                sdf_pred[:,1:] = sdf_pred[:,1:] * 255

                sdf_result_target[x, :, :, :] = np.reshape(sdf_pred[:,:].cpu(), [resolution, resolution, 4])


            # print('Minimum and maximum value: %f and %f. ' % (np.min(sdf_result_target[:,:,:,0]), np.max(sdf_result_target[:,:,:,0])))
            if(np.min(sdf_result_target[:,:,:,0]) < 0 and np.max(sdf_result_target[:,:,:,0]) > 0):
                vertices, faces = marching_cubes(sdf_result_target[:,:,:,0])
                colors_v = exctract_colors_v(vertices, sdf_result_target)
                colors_f = exctract_colors_f(colors_v, faces)
                off_file = '../../image2sdf/%s/%s_%d_target.off' %(output_dir, scene, j)
                write_off(off_file, vertices, faces, colors_f)
                # print('Wrote %s.' % off_file)
            else:
                print("surface level: 0, should be comprise in between the minimum and maximum value")
        

            sdf_validation = torch.tensor(np.reshape(sdf_result_for_error_computation,[resolution * resolution * resolution, 4]))
            sdf_target= torch.tensor(np.reshape(sdf_result_target_for_error_computation,[resolution * resolution * resolution, 4]))

            # assign weight of 0 for easy samples that are well trained
            threshold_precision = 1/resolution
            weight_sdf = ~((sdf_validation[:,0] > threshold_precision).squeeze() * (sdf_target[:,0] > threshold_precision).squeeze()) \
                * ~((sdf_validation[:,0] < -threshold_precision).squeeze() * (sdf_target[:,0] < -threshold_precision).squeeze())

            # Compute l1 loss, only for samples close to the surface
            loss_sdf = torch.nn.L1Loss(reduction='none')(sdf_validation[:,0].squeeze(), sdf_target[:,0])
            loss_sdf = (loss_sdf * weight_sdf).mean() * weight_sdf.numel()/weight_sdf.count_nonzero()
            loss_sdf *= resolution
        
            # loss rgb
            rgb_gt_normalized = sdf_target[:,1:]
            loss_rgb = torch.nn.L1Loss(reduction='none')(sdf_validation[:,1:], rgb_gt_normalized)
            loss_rgb = ((loss_rgb[:,0] * weight_sdf) + (loss_rgb[:,1] * weight_sdf) + (loss_rgb[:,2] * weight_sdf)).mean()/3 * weight_sdf.numel()/weight_sdf.count_nonzero()
            loss_rgb *= 255

            print(f"sdf loss: {loss_sdf}")
            print(f"rgb loss: {loss_rgb}")


    # similarity_same_model_cos = []
    # similarity_different_model_cos = []

    # similarity_same_model_l2 = []
    # similarity_different_model_l2 = []

    # for scene_id_1 in range(num_scene):
    #     for scene_id_2 in range(scene_id_1, num_scene):
    #         for vec1 in range(num_image_per_scene):
    #             for vec2 in range(num_image_per_scene):
    #                 dist = cosine_distance(lat_vecs[scene_id_1,vec1,:], lat_vecs[scene_id_2,vec2,:])
    #                 l2 = torch.norm(lat_vecs[scene_id_1,vec1,:]- lat_vecs[scene_id_2,vec2,:])
    #                 if scene_id_1 == scene_id_2 and vec2 != vec1:
    #                     similarity_same_model_cos.append(dist)
    #                     similarity_same_model_l2.append(l2)
    #                 elif scene_id_1 != scene_id_2:
    #                     similarity_different_model_cos.append(dist)
    #                     similarity_different_model_l2.append(l2)


    # print(f"average similarity between same models cosinus: {torch.tensor(similarity_same_model_cos).mean()}")
    # print(f"average similarity between differents models cosinus: {torch.tensor(similarity_different_model_cos).mean()}")

    # print(f"average similarity between same models l2: {torch.tensor(similarity_same_model_l2).mean()}")
    # print(f"average similarity between differents models l2: {torch.tensor(similarity_different_model_l2).mean()}")


    # matrix_cos_dist = np.empty([num_scene * num_image_per_scene, num_scene * num_image_per_scene])
    # matrix_l2_dist = np.empty([num_scene * num_image_per_scene, num_scene * num_image_per_scene])

    # for scene_id_1 in range(num_scene):
    #     for scene_id_2 in range(num_scene):
    #         for vec1 in range(num_image_per_scene):
    #             for vec2 in range(num_image_per_scene):
    #                 dist = cosine_distance(lat_vecs[scene_id_1,vec1,:], lat_vecs[scene_id_2,vec2,:])
    #                 matrix_cos_dist[scene_id_1 * num_image_per_scene + vec1, scene_id_2 * num_image_per_scene + vec2] = dist
    #                 matrix_l2_dist[scene_id_1 * num_image_per_scene + vec1, scene_id_2 * num_image_per_scene + vec2] = torch.norm(lat_vecs[scene_id_1,vec1,:] - lat_vecs[scene_id_2,vec2,:])

    # matrix_l2_dist = matrix_l2_dist/matrix_l2_dist.mean()


    # plt.figure()
    # plt.imshow(matrix_cos_dist, cmap='RdBu')
    # plt.title("cosine distance")
    # plt.colorbar()
    # if args.type == 'grid':
    #     plt.savefig("../../image2sdf/logs/encoder_grid/cosine_distance.png")
    # else:
    #     plt.savefig("../../image2sdf/logs/encoder_face/cosine_distance.png")

 
    # clustered_cosine_dist, idx = cluster_corr(matrix_cos_dist)
    # plt.figure()
    # plt.imshow(clustered_cosine_dist, cmap = 'RdBu')
    # plt.title("cosine distance")
    # plt.colorbar()
    # if args.type == 'grid':
    #     plt.savefig("../../image2sdf/logs/encoder_grid/cosine_distance_clustered.png")
    # else:
    #     plt.savefig("../../image2sdf/logs/encoder_face/cosine_distance_clustered.png")



    # plt.figure()
    # plt.imshow(matrix_l2_dist)
    # plt.title("l2 distance")
    # plt.colorbar()
    # if args.type == 'grid':
    #     plt.savefig("../../image2sdf/logs/encoder_grid/l2_distance.png")
    # else:
    #     plt.savefig("../../image2sdf/logs/encoder_face/l2_distance.png")

 
    # clustered_l2_dist, idx = cluster_corr(matrix_l2_dist)
    # plt.figure()
    # plt.imshow(clustered_l2_dist)
    # plt.title("l2 distance")
    # plt.colorbar()
    # if args.type == 'grid':
    #     plt.savefig("../../image2sdf/logs/encoder_grid/l2_distance_clustered.png")
    # else:
    #     plt.savefig("../../image2sdf/logs/encoder_face/l2_distance_clustered.png")