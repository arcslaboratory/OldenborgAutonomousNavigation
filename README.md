# Oldenborg Autonomous Navigation

Code for training machine learning models to navigate our Oldenborg simulation environment.

- `ue5osc`: Code to communicate between Unreal Engine 5 (or packaged simulations) and Python
- `boxnav`: Code to compute "correct" actions based on the agent's location in simulation
- `learning`: Code to train machine learning models to navigate the simulation


## Workflow

1. Generate data using `boxnav/boxsim.py`
2. Upload data using `learning/upload_data.py`.
3. Train model using `learning/training.py`.
4. Perform inference using `learning/inference.py`.

For example,

~~~bash
# Activate the environment
conda activate ENVIRONMENT

# Runs a navigator in Python and the simulation environment
# This will run on a system that can run Unreal Engine
# Open unreal engine before running code (ie. ARCSAssets.exe)
    # This will steal your mouse, run on split screen and use Alt+tab to navigate between screens 
cd boxnav
python boxsim.py NAVIGATOR --save_images IMAGE_DIRECTORY

# Uploads the dataset to the server
# You can upload from wherever the data is generated (probably the same system as above)
# For first time: Terminal will ask you to log-in to wandb 
    # *When asked to authorize, authentication code will be hidden when pasted so hit
    # ctrl+v once and hit enter
cd learning
python upload_data.py DATA_NAME PROJECT_NAME "Sample description of uploading run..." IMAGE_DIRECTORY

# Trains the model
# This should be run on a system with a GPU (e.g., our server)
cd learning
python training.py MODEL_NAME PROJECT_NAME "Sample description of training run..." ARCHITECTURE_NAME DATA_NAME(S)

# Performs inference
# This will run on a system that can run Unreal Engine
# Note: MODEL_NAME_FROM_WANDB can be found in arcslaboratory -> Projects -> PROJECT_NAME -> Artifacts
cd learning
python inference.py INFERENCE_NAME PROJECT_NAME "Sample description of inference run..." MODEL_NAME_FROM_WANDB IMAGE_SAVE_FOLDER_NAME
~~~

Notice that the first program argument for uploading, training, and inference is the name of the artifact created by that step.
