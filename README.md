# Context
This project is the result of a Master Thesis in the company Invision AI, in collaboration with CVLab at EPFL

We aim at modeling the shape and appearance of a vehicle detected by a tracker that predicts 3D bounding box associated to each detected vehicle.

One can find the report in this folder.

## Creating the environment
create the python virutal environment with ./code_neural_networks/create_venv.sh

## Initializing folders
Init the result folder with ./init_results_folder.sh

Add the SDF files of the models that you want to use for training, then update the "vehicle_list_all.txt" file in "code_neural_networks/config/"

## Generate training images
open synthesize_images.py with a text editor and update the path of DEFAULT_SHAPENET_DIRECTORY and DEFAULT_DIRECTORY.

run blender -b -P synthesize_images.py to generate the input images for the encoder.

## Training the networks
Now your ready for training the network, you can run:
- python code_neural_networks/scripts/decoderTraining.py
- pyhton code_neural_networks/scripts/encoderTraining.py

## Visualize results
Finally, you can compute the results with the various "evaluations" scripts in code_neural_networks/scripts and then visualize them in the results folder.
