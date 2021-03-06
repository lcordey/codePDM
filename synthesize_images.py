### NEED TO BE RUN WITH: blender -b -P synthesize_images.py -- --mode <training or validation> --venv <path/to/venv/dir>
###--shapenet_path <path/to/shapenent/dir> --output_path <path/to/output/synthesized_images> 


from typing import Dict
import bpy
import bpy_extras
import os
import sys
import argparse
import glob
import math
import numpy as np
import tempfile
import yaml
import pickle
import time

from random import random, randrange, choice

DEFAULT_SHAPENET_DIRECTORY = '/home/loic/data/vehicle'
DEFAULT_DIRECTORY = ''

DEFAULT_RESULTS_DIRECTORY = DEFAULT_DIRECTORY + 'results/'
DEFAUT_VEHICLE_LIST_PATH = DEFAULT_DIRECTORY + 'code_neural_networks/config/vehicle_list_all.txt'

DEFAULT_MODE = 'training'
NUM_SCENE_TRAINING = 100
NUM_SCENE_VALIDATION = 100

# import IPython

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, dest='mode', default= DEFAULT_MODE, help='training or validation')
parser.add_argument('--venv', dest='venv_path', default='/home/loic/venvs/blender/lib/python3.7/site-packages',
                    help = 'path to the site-packages folder of the virtual environment')
parser.add_argument('--shapenet_path', dest='shapenet_path', default=DEFAULT_SHAPENET_DIRECTORY,
                    help='path to the dataset')
parser.add_argument('--directory_path', dest='directory_path', default=DEFAULT_RESULTS_DIRECTORY, help='path to the folder directory')
parser.add_argument('--height', type=int, dest='height', default=300)
parser.add_argument('--width', type=int, dest='width', default=300)

args = parser.parse_args(args=sys.argv[5:])

sys.path.append(args.venv_path)

import IPython
import mathutils
from mathutils import *
print(f'\nPath to bpy: {bpy.__file__}\n')
from PIL import Image

def init_blender():
    """ create initial blender scene"""
    scene = bpy.context.scene

    bpy.context.scene.render.film_transparent = True

    scene.render.resolution_x = 300
    scene.render.resolution_y = 300
    scene.render.resolution_percentage = 100

    objs = bpy.data.objects
    objs.remove(objs["Cube"])

    # tag existing items
    for obj in bpy.data.objects:
        obj.tag = True

def load_model(model_name: str):
    """import vehicle model and set its name"""
    if os.path.isfile(model_name):
        bpy.ops.import_scene.obj(filepath=model_name)
    else:
        bpy.ops.object.add(type='EMPTY')

    obj_list = [obj for obj in bpy.data.objects if obj.tag is False]
    # set name of untaged items

    for obj in obj_list:
        obj.select_set(True)
        obj.name = 'model'

def get_bounding_box() -> list:
    """returns the pixel values of the 6 points needed for the bounding box reconstruction"""
    scene = bpy.context.scene

    # needed to rescale 2d coordinates
    render = scene.render
    res_x = render.resolution_x
    res_y = render.resolution_y

    cam = bpy.data.objects['Camera']
    obj = bpy.data.objects['model']

    bbox = obj.bound_box
    # find minimum and maximum of each coordinate
    min_coord = np.min([v[:] for v in bbox], axis=0)
    max_coord = np.max([v[:] for v in bbox], axis=0)

    # get 3d position of relevant points (y and z axis are inverted)
    p0 = mathutils.Vector((max_coord[0], min_coord[1], min_coord[2]))
    p1 = mathutils.Vector((min_coord[0], min_coord[1], min_coord[2]))
    p2 = mathutils.Vector((min_coord[0], max_coord[1], min_coord[2]))
    p3 = mathutils.Vector((max_coord[0], max_coord[1], min_coord[2]))
    p4 = mathutils.Vector((max_coord[0], min_coord[1], max_coord[2]))
    p5 = mathutils.Vector((min_coord[0], min_coord[1], max_coord[2]))
    p6 = mathutils.Vector((min_coord[0], max_coord[1], max_coord[2]))
    p7 = mathutils.Vector((max_coord[0], max_coord[1], max_coord[2]))

    points = [p0, p1, p2, p3, p4, p5, p6, p7]

    points = [obj.matrix_world @ p for p in points]

    # get 2d placements of the points in the render
    points_2d = [bpy_extras.object_utils.world_to_camera_view(scene, cam, p) for p in points]


    ### rescaling to get pixel values
    # for p in points_2d:
    #     p.x = p.x*res_x
    #     p.y = res_y - p.y*res_y

    return points_2d, points


def randomize_vehicle_placement(temp_filepath: str):
    obj = bpy.data.objects['model']

    r_scale = 6
    r_rot = random()
    r_x = 0
    r_y = 0

    obj.location = (r_x, r_y, 0)
    obj.scale = (r_scale, r_scale, r_scale)
    obj.rotation_euler.rotate_axis("Y", math.radians(2*r_rot * 180))
    # file needs to be saved so that the bounding box information is updated
    bpy.ops.wm.save_as_mainfile(filepath=temp_filepath)


def set_fixed_vehicle_placement(temp_filepath: str):
    obj = bpy.data.objects['model']
    r_scale = 1
    # r_scale = 6
    # r_rot = 0.0

    obj.location = (0, 0, 0)
    obj.scale = (r_scale, r_scale, r_scale)
    # obj.rotation_euler.rotate_axis("Y", math.radians(2 * r_rot * 180))
    # file needs to be saved so that the bounding box information is updated
    bpy.ops.wm.save_as_mainfile(filepath=temp_filepath)

def save_pickle_file(image_dict, pickle_file_path):
    output_file = open(pickle_file_path, "wb")
    pickle.dump(image_dict, output_file)
    output_file.close()

def render_to_file(rendered_image_path: str):
    """render the image"""

    bpy.context.scene.render.filepath = rendered_image_path
    bpy.ops.render.render(write_still = True)


def get_shape_dirs(path: str, vehicle_list: set) -> list:
    def is_listed(model_dir: str) -> bool:
        shape_hash = os.path.split(model_dir)[-1]
        return shape_hash in vehicle_list

    return list(filter(is_listed, glob.glob(f'{path}/*')))


def generate_white_background(w: int, h: int) -> Image:
    return Image.new('RGB', (w, h), (255, 255, 255))


def update_camera(camera, focus_point=mathutils.Vector((0.0, 0.0, 0.0)), distance=2.0):
    """
    Focus the camera to a focus point and place the camera at a specific distance from that
    focus point. The camera stays in a direct line with the focus point.

    :param camera: the camera object
    :type camera: bpy.types.object
    :param focus_point: the point to focus on (default=``mathutils.Vector((0.0, 0.0, 0.0))``)
    :type focus_point: mathutils.Vector
    :param distance: the distance to keep to the focus point (default=``10.0``)
    :type distance: float
    """
    looking_direction = camera.location - focus_point
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()
    # Use * instead of @ for Blender <2.8
    camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))


assert(args.mode == 'training' or args.mode == 'validation'), "please give either training or validation as mode"

#create folder and annotation file
assert os.path.isdir(args.directory_path), f"main directory does not exists: {args.directory_path}"

if args.mode == 'training':
    output_path = args.directory_path + 'input_images'
    vehicle_list_path = DEFAUT_VEHICLE_LIST_PATH
    num_scenes_per_vehicule = NUM_SCENE_TRAINING
else:
    output_path = args.directory_path + 'input_images_validation'
    vehicle_list_path = DEFAUT_VEHICLE_LIST_PATH
    num_scenes_per_vehicule = NUM_SCENE_VALIDATION


if not os.path.isdir(output_path) :
    os.mkdir(output_path)

if not os.path.isdir(f'{output_path}/images') :
    os.mkdir(f'{output_path}/images')
    
annotations_path = f'{output_path}/annotations.pkl'

w, h = args.width, args.height

init_blender()

with open(vehicle_list_path) as f:
    vehicle_list = f.read().splitlines()

vehicle_list = set(vehicle_list)
vehicle_pool = get_shape_dirs(args.shapenet_path, vehicle_list)


# update camera location
bpy.data.objects['Camera'].location.x = 1
bpy.data.objects['Camera'].location.y = 0
bpy.data.objects['Camera'].location.z = 1
# update camera orientation and distance
update_camera(bpy.data.objects['Camera'])

#temporary files used to save the blender scene
init_temp_file = tempfile.NamedTemporaryFile()
temp_file = tempfile.NamedTemporaryFile()

#save initial scene
bpy.ops.wm.save_as_mainfile(filepath=init_temp_file.name)


annotations = dict()

time_start = time.time()

for i in range(len(vehicle_pool)):

    model_dir = vehicle_pool[i]
    model_hash = os.path.split(model_dir)[-1]
    annotations[model_hash] = dict()

    load_model(f'{model_dir}/models/model_normalized.obj')
    set_fixed_vehicle_placement(temp_file.name)

    obj = bpy.data.objects['model']
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'

    for j in range(num_scenes_per_vehicule):
        
        # # rotate the car randomly
        # if j != 0:
        #     obj.rotation_euler.rotate_axis("Y", math.radians(2 * random() * 180)) 

        # # add a random rotation to simulate different height of camera 
        # if j != 0:
        #     rz = (random() - 0.5) * 2
        #     obj.rotation_euler = (mathutils.Matrix.Rotation(math.pi/6 * rz, 3, 'Y') @ obj.rotation_euler.to_matrix()).to_euler()


        if j == 0:
            obj.rotation_euler = (mathutils.Matrix.Rotation(-math.pi/4, 3, 'Y') @ obj.rotation_euler.to_matrix()).to_euler()

        elif j == 1:
            obj.rotation_euler = (mathutils.Matrix.Rotation(math.pi/4, 3, 'Y') @ obj.rotation_euler.to_matrix()).to_euler()

        elif j == 2:
            obj.rotation_euler.rotate_axis("Y", math.radians(-90)) 
            obj.rotation_euler = (mathutils.Matrix.Rotation(-math.pi/4, 3, 'Y') @ obj.rotation_euler.to_matrix()).to_euler()

        else:
            obj.rotation_euler.rotate_axis("Y", math.radians(2 * random() * 180)) 
            rz = (random() - 0.5) * 2
            # obj.rotation_euler = (mathutils.Matrix.Rotation(math.pi/6 * rz, 3, 'Y') @ obj.rotation_euler.to_matrix()).to_euler()
            obj.rotation_euler = (mathutils.Matrix.Rotation(math.pi/3 * rz, 3, 'Y') @ obj.rotation_euler.to_matrix()).to_euler()

        rendered_image_path = f'images/{model_hash}/{j}.png'
        render_to_file(f'{output_path}/{rendered_image_path}')
        points_2d, points_3d = get_bounding_box()

        annotations[model_hash][j] = dict()
        annotations[model_hash][j]['2d'] = np.array(points_2d)
        annotations[model_hash][j]['3d'] = np.array(points_3d)
        annotations[model_hash][j]['frame'] = [v for v in np.array(bpy.data.objects['Camera'].data.view_frame(scene=bpy.context.scene)[:3])]
        annotations[model_hash][j]['matrix_object_to_world'] = np.array(bpy.data.objects['model'].matrix_world)

        # load the image and apply background if needed
        img = Image.open(f'{output_path}/{rendered_image_path}')
        background_image_crop = generate_white_background(w, h)
        background_image_crop.paste(img, (0, 0), img)
        background_image_crop.save(f'{output_path}/{rendered_image_path}', "PNG")

        # # replace the camera to the original orientation
        # if j != 0:
        #     obj.rotation_euler = (mathutils.Matrix.Rotation(-math.pi/6 * rz, 3, 'Y') @ obj.rotation_euler.to_matrix()).to_euler()


        if j == 0:
            obj.rotation_euler = (mathutils.Matrix.Rotation(+math.pi/4, 3, 'Y') @ obj.rotation_euler.to_matrix()).to_euler()

        elif j == 1:
            obj.rotation_euler = (mathutils.Matrix.Rotation(-math.pi/4, 3, 'Y') @ obj.rotation_euler.to_matrix()).to_euler()

        elif j == 2:
            obj.rotation_euler.rotate_axis("Y", math.radians(+90)) 
            obj.rotation_euler = (mathutils.Matrix.Rotation(+math.pi/4, 3, 'Y') @ obj.rotation_euler.to_matrix()).to_euler()

        else:
            # obj.rotation_euler = (mathutils.Matrix.Rotation(-math.pi/6 * rz, 3, 'Y') @ obj.rotation_euler.to_matrix()).to_euler()
            obj.rotation_euler = (mathutils.Matrix.Rotation(-math.pi/3 * rz, 3, 'Y') @ obj.rotation_euler.to_matrix()).to_euler()


    # reload the scene to reduce the memory leak issue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=init_temp_file.name)
    for obj in bpy.data.objects:
        obj.tag = True

matrix_world_to_camera = np.array(bpy.data.objects['Camera'].matrix_world.normalized().inverted())
annotations["matrix_world_to_camera"] = matrix_world_to_camera

print(f"rendering time: {time.time() - time_start}")
time_start = time.time()

save_pickle_file(annotations, annotations_path)

print(f"saving time: {time.time() - time_start}")

# IPython.embed()
