### NEED TO BE RUN WITH: blender -b -P synthesize_images.py -- --venv <path/to/venv/dir>
###--shapenet_path <path/to/shapenent/dir> --output_path <path/to/output/synthesized_images> 


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

from random import random, randrange, choice

DEFAULT_MODE = 'validation'
NUM_SCENE_TRAINING = 30
NUM_SCENE_VALIDATION = 3

# import IPython

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, dest='mode', default= DEFAULT_MODE, help='training or validation images')
parser.add_argument('--venv', dest='venv_path', default='/home/loic/venvs/blender/lib/python3.7/site-packages',
                    help = 'path to the site-packages folder of the virtual environment')
parser.add_argument('--shapenet_path', dest='shapenet_path', default='/home/loic/data/vehicle',
                    help='path to the dataset')
parser.add_argument('--directory_path', dest='directory_path', default='/home/loic/MasterPDM/image2sdf/', help='path to the folder directory')
parser.add_argument('--height', type=int, dest='height', default=300)
parser.add_argument('--width', type=int, dest='width', default=450)
# parser.add_argument('--num_images', type=int, dest='num_images', default=100, help='number of output images')
# parser.add_argument('--image_name', dest='image_name', default='synth_image', help='name of the output images')

args = parser.parse_args(args=sys.argv[5:])

sys.path.append(args.venv_path)

import IPython
import mathutils
from PIL import Image

def init_blender():
    """ create initial blender scene"""
    scene = bpy.context.scene

    bpy.context.scene.render.film_transparent = True

    scene.render.resolution_x = 450
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


def generate_image_dict(rendered_image_path: str, points_2d: list) -> dict:
    """ write bounding box information in pkl file"""
    image_dict = dict()

    for pid in range(8):
        image_dict[f'p{pid + 1}x'] = points_2d[pid].x
        image_dict[f'p{pid + 1}y'] = points_2d[pid].y
    
    # 2d bounding box values
    min_px = min(p.x for p in points_2d)
    min_py = min(p.y for p in points_2d)
    max_px = max(p.x for p in points_2d)
    max_py = max(p.y for p in points_2d)

    image_dict['rect_height'] = max_py - min_py
    image_dict['rect_width'] = max_px - min_px
    image_dict['rect_x'] = min_px
    image_dict['rect_y'] = min_py


    return image_dict

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
    p4 = mathutils.Vector((min_coord[0], min_coord[1], max_coord[2]))
    p5 = mathutils.Vector((min_coord[0], max_coord[1], max_coord[2]))
    p6 = mathutils.Vector((max_coord[0], min_coord[1], max_coord[2]))
    p7 = mathutils.Vector((max_coord[0], max_coord[1], max_coord[2]))

    points = [p0, p1, p2, p3, p4, p5, p6, p7]

    points = [obj.matrix_world @ p for p in points]

    # get 2d placements of the points in the render
    points_2d = [bpy_extras.object_utils.world_to_camera_view(scene, cam, p) for p in points]
    # rescaling to get pixel values
    for p in points_2d:
        p.x = p.x*res_x
        p.y = res_y - p.y*res_y

    return points_2d


def randomize_vehicle_placement(temp_filepath: str):
    obj = bpy.data.objects['model']

    r_scale = 6
    # r_rot = (int)(random() * 18) /18
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
    r_scale = 6
    r_rot = 0.5

    obj.location = (0, 0, 0)
    obj.scale = (r_scale, r_scale, r_scale)
    obj.rotation_euler.rotate_axis("Y", math.radians(2 * r_rot * 180))
    # file needs to be saved so that the bounding box information is updated
    bpy.ops.wm.save_as_mainfile(filepath=temp_filepath)


def save_yaml_file(image_dict_list: list, yaml_path: str):
    #save current list
    with open(yaml_path, 'w') as yamlfile:
        yaml.dump(image_dict_list, yamlfile)

def save_pickle_file(image_dict, pickle_file_path):
    output_file = open(pickle_file_path, "wb")
    pickle.dump(image_dict, output_file)
    output_file.close()

def render_to_file(rendered_image_path: str):
    """render the image and delete the current vehicle"""
    bpy.context.scene.render.filepath = rendered_image_path
    bpy.ops.render.render(write_still = True)


def get_shape_dirs(path: str, whitelist: set) -> list:
    def is_whitelisted(model_dir: str) -> bool:
        shape_hash = os.path.split(model_dir)[-1]
        return shape_hash in whitelist

    return list(filter(is_whitelisted, glob.glob(f'{path}/*')))


def generate_white_background(w: int, h: int) -> Image:
    return Image.new('RGB', (w, h), (255, 255, 255))


assert(args.mode == 'training' or args.mode == 'validation'), "please give either training or validation as mode"

#create folder and annotation file
assert os.path.isdir(args.directory_path), f"main directory does not exists: {args.directory_path}"

if args.mode == 'training':
    output_path = args.directory_path + 'input_images'
    whitelist_path = '/home/loic/MasterPDM/image2sdf/vehicle_whitelist.txt'
    num_scenes_per_vehicule = NUM_SCENE_TRAINING
else:
    output_path = args.directory_path + 'input_images_validation'
    whitelist_path = '/home/loic/MasterPDM/image2sdf/vehicle_whitelist_validation.txt'
    num_scenes_per_vehicule = NUM_SCENE_VALIDATION

if not os.path.isdir(f'{output_path}/images') :
    os.mkdir(f'{output_path}/images')

yaml_file_path = f'{output_path}/annotations.yaml'
pickle_file_path = f'{output_path}/annotations.pkl'

w, h = args.width, args.height

init_blender()

#temporary files used to save the blender scene
init_temp_file = tempfile.NamedTemporaryFile()
temp_file = tempfile.NamedTemporaryFile()
#save initial scene
bpy.ops.wm.save_as_mainfile(filepath=init_temp_file.name)

    
with open(whitelist_path) as f:
    whitelisted_vehicles = f.read().splitlines()

whitelisted_vehicles = set(whitelisted_vehicles)
vehicle_pool = get_shape_dirs(args.shapenet_path, whitelisted_vehicles)

image_dict = dict()

for i in range(len(vehicle_pool)):
    model_dir = vehicle_pool[i]
    model_id = os.path.split(model_dir)[-1]

    image_dict[model_id] = []


    load_model(f'{model_dir}/models/model_normalized.obj')
    set_fixed_vehicle_placement(temp_file.name)

    obj = bpy.data.objects['model']

    for j in range(num_scenes_per_vehicule):

        if args.mode == 'training':
            # set_fixed_vehicle_placement(temp_file.name, j/num_scenes_per_vehicule)
            obj.rotation_euler.rotate_axis("Y", math.radians(2 * 1/num_scenes_per_vehicule * 180))
        else:
            # randomize_vehicle_placement(temp_file.name)
            obj.rotation_euler.rotate_axis("Y", math.radians(2 * random() * 180))

        points_2d = get_bounding_box()
        rendered_image_path = f'images/{model_id}/{j}.png'
        image_dict[model_id].append(generate_image_dict(rendered_image_path, points_2d))
        render_to_file(f'{output_path}/{rendered_image_path}')


        # load the image and apply background if needed
        img = Image.open(f'{output_path}/{rendered_image_path}')
        background_image_crop = generate_white_background(w, h)
        background_image_crop.paste(img, (0, 0), img)
        background_image_crop.save(f'{output_path}/{rendered_image_path}', "PNG")

    # reload the scene to reduce the memory leak issue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=init_temp_file.name)
    for obj in bpy.data.objects:
        obj.tag = True

save_yaml_file(image_dict, yaml_file_path)
save_pickle_file(image_dict, pickle_file_path)



# IPython.embed()
