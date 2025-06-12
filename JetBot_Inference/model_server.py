# Don't Change Order of Imports to avoid the error "cannot allocate memory in static TLS block"
import pathlib
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
import wandb
from fastai.callback.wandb import WandbCallback
from fastai.learner import load_learner
from rpc import RPCServer
from utils import y_from_filename  # noqa: F401 (needed for fastai load_learner)

# Initialize the RPC server
server = RPCServer()

@contextmanager
def set_posix_windows():
    # Workaround for fastai/pathlib issue on Windows by temporarily replacing PosixPath with WindowsPath
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup

def parse_args():
    # Parse command-line arguments for wandb configuration and model path
    arg_parser = ArgumentParser("Track performance of trained networks.")
    arg_parser.add_argument("--wandb_name", default="jetbot_test_v8_aser", help="Name of run inference results.")
    arg_parser.add_argument("--wandb_project", default="Summer2024Official", help="Wandb project name.")
    arg_parser.add_argument("--wandb_notes", default="test of jetbot using rpc server", help="Wandb run description.")
    arg_parser.add_argument("--wandb_model", default="WanderingRand50ResNet18-ResNet18-Wandering100kRandEvery50Data-rep00", help="Path to the model to evaluate.")
    return arg_parser.parse_args()

args = parse_args()
wandb_entity = "arcslaboratory"
wandb_project = args.wandb_project
wandb_name = args.wandb_name
wandb_notes = args.wandb_notes
wandb_model = args.wandb_model

# Initialize a Weights & Biases (wandb) run for logging inference results
run = wandb.init(
    job_type="inference",
    entity=wandb_entity,
    name=wandb_name,
    project=wandb_project,
    notes=wandb_notes,
)

# Ensure the wandb run was successfully initialized
if run is None:
    raise Exception("wandb.init() failed")

# Download the model artifact from wandb
artifact = run.use_artifact(f"{wandb_model}:latest", type="model")
model_dir = artifact.download()
model_filename = Path(model_dir) / wandb_model
model_filename = str(model_filename) + ".pkl"
model_filename = Path(model_filename)

# Load the trained model using fastai's load_learner
model = load_learner(model_filename)

# Remove WandbCallback from the model (temporary fix)
model.remove_cb(WandbCallback)

def model_run(image_filename):
    # Perform inference on the given image and return the predicted action
    action_to_take, action_index, action_probs = model.predict(image_filename)
    action_prob = action_probs[action_index]
    print(action_prob)
    print(" " + action_to_take + " \n")
    return action_to_take

# Register the model_run function as an RPC method
server.registerMethod(model_run)

# Start the RPC server
server.run()
