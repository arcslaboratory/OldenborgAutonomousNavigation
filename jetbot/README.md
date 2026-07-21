# JetBot Inference

The JetBot SDK uses an older, incompatible version of Python when compared to the latest versions of PyTorch and fastai. To overcome this limitation, we must create two separate Python environments: one for JetBot control and another for machine learning. We communicate between these environments using Remote Procedure Calls (RPCs).

The JetBot captures an image, saves it locally, and calls the model inference function with the image filename. The model loads the image, runs the inference, outputs a motion direction, and then calls the robot motion function with that direction.

`rpcs.py` This file contains the RPC server and RPC client classes. The RPC server is responsible for starting the server, receiving RPC calls from the client, executing the requested functions, and sending back the result. The RPC client is responsible for connecting to the server and making RPC calls to the server.

`model_server.py` This file starts the RPC server and is responsible for downloading the model from WandB, loading the model, running the inference, and returning the motion direction.

```bash
cd /home/jetbot/arcs/OldenborgAutonomousNavigation/jetbot

mamba activate arcs-su25

# Run the model server using the conda Python interpreter
python model_server.py --wandb_name "jetbot-test" --wandb_project "summer2025-del" --wandb_model "test1-ResNet18-test-rep00" --wandb-notes "Testing the model server."
```

`robot_client.py` This file starts the RPC client and is responsible for capturing an image, un-distorting it, saving it locally, calling the model inference function with the image filename, and then calling the robot motion function with the motion direction.

The robot client currently relies on the calibration images. This will need to change, but in the meantime, you'll need to copy the calibration images from our box folder (`/Users/ajcd2020/Library/CloudStorage/Box-Box/_mine/ARCSStorage/ARCSCloud/Images/JetBot1Calibration.zip`) to `camera_calibration/images`.

```bash
# In a separate terminal
cd /home/jetbot/arcs/OldenborgAutonomousNavigation/jetbot

# Run the robot client using the system Python3 interpreter
/usr/bin/python3 robot_client.py
```
