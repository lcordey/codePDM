create the python virutal environment with ./code_neural_networks/create_venv.sh

Init the result folder with ./init_results_folder.sh

Add the SDF files of the models that you want to use for training, then update the "vehicle_list_all.txt" file in "code_neural_networks/config/"

open synthesize_images.py with a text editor and update the path of DEFAULT_SHAPENET_DIRECTORY and DEFAULT_DIRECTORY.

run blender -b -P synthesize_images.py to generate the input images for the encoder.

Now your ready for training the network, you can run:
- python code_neural_networks/scripts/decoderTraining.py
- pyhton code_neural_networks/scripts/encoderTraining.py

Finally, you can compute the results with the various "evaluations" scripts in code_neural_networks/scripts and then visualize them in the results folder.
