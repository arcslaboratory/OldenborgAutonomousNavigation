"""
Use batch_tfms=aug_transforms() to apply data augmentation
Better for sim2real?
"""

from argparse import ArgumentParser, Namespace
from functools import partial
from math import radians
from pathlib import Path

# TODO: log plots as artifacts?
# import matplotlib.pyplot as plt
import torch
import wandb
from fastai.callback.wandb import WandbCallback
from fastai.data.all import (
    CategoryBlock,
    DataBlock,
    DataLoaders,
    RandomSplitter,
    RegressionBlock,
)
from fastai.losses import CrossEntropyLossFlat
from fastai.vision.augment import Resize
from fastai.vision.data import ImageBlock, ImageDataLoaders
from fastai.vision.learner import (
    Learner,
    accuracy,
    create_timm_model,
    default_split,
    vision_learner,
)
from fastai.vision.utils import get_image_files
from torch import nn


def parse_args() -> Namespace:
    arg_parser = ArgumentParser("Train command classification networks.")

    # Wandb configuration
    arg_parser.add_argument("wandb_name", help="Name of run and trained model.")
    arg_parser.add_argument("wandb_project", help="Wandb project name.")
    arg_parser.add_argument("wandb_notes", help="Wandb run description.")

    # Model configuration
    arg_parser.add_argument("model_arch", help="Model architecture (see code).")
    arg_parser.add_argument(
        "--use_command_image",
        action="store_true",
        help="Use the command+image input model.",
    )

    # Dataset configuration
    arg_parser.add_argument("dataset_names", nargs="+", help="Name of datasets to use.")
    arg_parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained model."
    )
    arg_parser.add_argument("--gpu", type=int, default=0, help="GPU to use.")
    arg_parser.add_argument(
        "--valid_pct", type=float, default=0.2, help="Validation percentage."
    )
    arg_parser.add_argument(
        "--rotation_threshold",
        type=float,
        default=radians(5),
        help="Threshold in radians for classifying rotation as left/right or forward.",
    )
    arg_parser.add_argument(
        "--local_data", action="store_true", help="Data is stored locally."
    )

    # Training configuration
    arg_parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs."
    )
    arg_parser.add_argument(
        "--num_replicates", type=int, default=1, help="Number of replicates to run."
    )
    arg_parser.add_argument(
        "--image_resize",
        type=int,
        default=244,
        help="The size of image training data.",
    )
    arg_parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size."
    )

    return arg_parser.parse_args()


def setup_wandb(args: Namespace):
    wandb_entity = "arcslaboratory"
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name
    wandb_notes = args.wandb_notes

    run = wandb.init(
        job_type="train",
        entity=wandb_entity,
        name=wandb_name,
        project=wandb_project,
        notes=wandb_notes,
    )

    if run is None:
        raise Exception("wandb.init() failed")

    data_dirs = []

    for dataset_name in args.dataset_names:
        if args.local_data:
            data_dir = dataset_name
        else:
            artifact = run.use_artifact(f"{dataset_name}:latest")
            data_dir = artifact.download()
        data_dirs.append(data_dir)

    return run, data_dirs


def get_angle_from_filename(filename: str) -> float:
    filename_stem = Path(filename).stem
    angle = float(filename_stem.split("_")[2].replace("p", "."))
    return angle


def y_from_filename(rotation_threshold: float, filename: str) -> str:
    """Extracts the direction label from the filename of an image.

    Example: "path/to/file/001_000011_-1p50.png" --> "right"
    """
    filename_stem = Path(filename).stem
    angle = float(filename_stem.split("_")[2].replace("p", "."))

    if angle > rotation_threshold:
        return "left"
    elif angle < -rotation_threshold:
        return "right"
    else:
        return "forward"


def get_dls(args: Namespace, data_paths: list):
    # NOTE: not allowed to add a type annotation to the input
    image_filenames = []
    for data_path in data_paths:
        image_filenames.extend(get_image_files(data_path))  # type:ignore

    # Using a partial function to set the rotation_threshold from args
    label_func = partial(y_from_filename, args.rotation_threshold)

    if args.use_command_image:
        return get_image_command_category_dataloaders(
            args, data_paths, image_filenames, label_func
        )
    else:
        return ImageDataLoaders.from_name_func(
            data_paths[0],  # TODO: find a better place to save models
            image_filenames,
            label_func,
            valid_pct=args.valid_pct,
            shuffle=True,
            bs=args.batch_size,
            item_tfms=Resize(args.image_resize),
        )


def get_image_command_category_dataloaders(
    args: Namespace, data_paths: list, image_filenames, y_from_filename
):
    def x1_from_filename(filename: str) -> str:
        return filename

    # NOTE: not allowed to add a type annotation to the input
    def x2_from_filename(filename) -> float:
        filename_index = image_filenames.index(Path(filename))

        if filename_index == 0:
            return 0.0

        previous_filename = image_filenames[filename_index - 1]
        previous_angle = get_angle_from_filename(previous_filename)

        if previous_angle > args.rotation_threshold:
            return 1.0
        elif previous_angle < -args.rotation_threshold:
            return 2.0
        else:
            return 0.0

    image_command_data = DataBlock(
        blocks=(ImageBlock, RegressionBlock, CategoryBlock),  # type: ignore
        n_inp=2,
        get_items=get_image_files,
        get_y=y_from_filename,
        get_x=[x1_from_filename, x2_from_filename],
        splitter=RandomSplitter(args.valid_pct),
        # item_tfms=Resize(args.image_resize),
    )

    # TODO: This is untested
    return image_command_data.dataloaders(
        data_paths[0], shuffle=True, batch_size=args.batch_size
    )


def run_experiment(args: Namespace, run, dls):
    torch.cuda.set_device(int(args.gpu))
    dls.to(torch.cuda.current_device())
    print("Running on GPU: " + str(torch.cuda.current_device()))

    learn = None
    for rep in range(args.num_replicates):
        learn = train_model(dls, args, run, rep)

    return learn


def train_model(dls: DataLoaders, args: Namespace, run, rep: int):
    """Train the cmd_model using the provided data and hyperparameters."""

    valid_architectures = [
        "ResNet18",
        "MobileNetV4",
        "EfficientNet",
        "ConvNextV2Atto",
        "ConvNextV2Base",
        "ViT",
    ]

    short_to_full_architectures = {
        "ResNet18": "resnet18.a1_in1k",
        "MobileNetV4": "mobilenetv4_conv_small.e2400_r224_in1k",
        "EfficientNet": "efficientnet_b3.ra2_in1k",
        "ConvNextV2Atto": "convnextv2_atto.fcmae",
        "ConvNextV2Base": "convnextv2_base.fcmae_ft_in22k_in1k",
        "ViT": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    }

    if args.model_arch not in valid_architectures:
        raise ValueError(f"Invalid model architecture: {args.model_arch}")

    if args.use_command_image:
        raise NotImplementedError("ImageCommandModel not implemented.")
        # net = ImageCommandModel(args.model_arch, pretrained=args.pretrained)
        # learn = Learner(
        #     dls=dls,
        #     model=net,
        #     loss_func=CrossEntropyLossFlat(),
        #     metrics=accuracy,
        #     cbs=WandbCallback(log_model=True),
        # )
    else:
        learn = vision_learner(
            dls,
            short_to_full_architectures[args.model_arch],
            pretrained=args.pretrained,
            metrics=accuracy,
            cbs=WandbCallback(log_model=True),
        )

    if args.pretrained:
        learn.fine_tune(args.num_epochs)
    else:
        learn.fit_one_cycle(args.num_epochs)

    wandb_name = args.wandb_name
    model_arch = args.model_arch
    dataset_names = "-".join(args.dataset_names)

    # store the datasets used in metadata
    # TODO: see if can store this in more accessible place
    wandb.log({"Datasets Used": str(args.dataset_names)})
    wandb.config.update({"Datasets Used": str(args.dataset_names)})

    if len(dataset_names) > 1:
        learn_name = f"{wandb_name}-{model_arch}-MultipleDatasets-rep{rep:02}"
    else:
        learn_name = f"{wandb_name}-{model_arch}-{dataset_names}-rep{rep:02}"

    learn_filename = learn_name + ".pkl"
    learn.export(learn_filename)

    learn_path = learn.path / learn_filename
    artifact = wandb.Artifact(name=learn_name, type="model")
    artifact.add_file(local_path=learn_path)
    run.log_artifact(artifact)


class ImageCommandModel(nn.Module):
    """Initializes the CommandModel class."""

    def __init__(self, architecture_name: str, pretrained: bool):
        super(ImageCommandModel, self).__init__()
        # cnn_constructor = compared_models[architecture_name]
        # weights = "IMAGENET1K_V1" if pretrained else None
        # self.cnn = cnn_constructor(weights=weights)
        model, cfg = create_timm_model(architecture_name, 3, default_split, pretrained)
        # _timm_norm(dls, cfg, pretrained, n_in=3)
        self.cnn = model

        # Layers to combine image and command input
        self.fc1 = nn.Linear(self.cnn.fc.out_features + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, img, cmd):
        """Performs a forward pass through the model."""

        # Pass the image data to the cnn
        x_image = self.cnn(img)

        # Returns a new tensor from the cmd data
        x_command = cmd.unsqueeze(1)

        # Concatenate cmd and image in the 1st dimension
        x = torch.cat((x_image, x_command), dim=1)

        # Apply the ReLU function element-wise to the linearly transformed img+cmd data
        x = self.r1(self.fc1(x))

        # Apply the linear transformation to the data
        x = self.fc2(x)

        # The loss function applies softmax to the output of the model
        return x


def main():
    args = parse_args()
    run, data_paths = setup_wandb(args)
    dls = get_dls(args, data_paths)
    run_experiment(args, run, dls)


if __name__ == "__main__":
    main()
