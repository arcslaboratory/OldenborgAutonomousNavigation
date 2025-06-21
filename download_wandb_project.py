import wandb
import os
import argparse

# example usage:
# python download_wandb_project.py Summer2024Official --output_dir Summer2024Official_downloads


def download_project_runs(project: str, output_dir: str):
    """
    Download all runs and their artifacts from a specified WandB project.

    Args:
        project (str): The name of the WandB project.
        output_dir (str): The directory to save downloaded runs and artifacts.
    """
    api = wandb.Api()  # Initialize the WandB API

    runs = api.runs(f"arcslaboratory/{project}")  # Get all runs for the project
    print(f"Found {len(runs)} runs in project '{project}'")

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    for run in runs:
        run_dir = os.path.join(output_dir, run.id)  # Directory for this run
        os.makedirs(run_dir, exist_ok=True)  # Create directory for the run

        print(f"\nDownloading run: {run.name} ({run.id})")

        # Download all files associated with the run
        for file in run.files():
            print(f"File: {file.name}")
            file.download(root=run_dir, replace=True)  # Download file to run_dir

        # Download all logged artifacts for the run
        for artifact in run.logged_artifacts():
            artifact_name = f"{artifact.name.replace('/', '_')}:{artifact.version}"  # Format artifact name
            print(f"Artifact: {artifact_name}")
            artifact.download(root=run_dir)  # Download artifact to run_dir


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Download all runs and artifacts from a WandB project."
    )
    parser.add_argument(
        "project", help="WandB project name"
    )  # Required project name argument
    parser.add_argument(
        "--output_dir",
        default="wandb_downloads",
        help="Directory to save the runs and artifacts",
    )

    args = parser.parse_args()  # Parse command-line arguments
    download_project_runs(args.project, args.output_dir)  # Run the download function
