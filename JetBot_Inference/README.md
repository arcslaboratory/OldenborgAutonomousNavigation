**Remote Procedure Calls (RPC)** <br/>
To overcome the incompatibility of newer PyTorch and Fast.ai versions that were used in model development with Python 3.6, which is the latest version supporting the Nvidia Jetbot, we built two separate environments: one for model execution and one for robot control, communicating via Remote Procedure Calls (RPCs). The Jetbot captures an image, saves it locally, and calls the model inference function with the image filename. The model loads the image, runs the inference, outputs a motion direction, and then calls the robot motion function with that direction.

**rpcs.py** <br/>
This file contains the RPC server and RPC client classes. The RPC server is responsible for starting the server, receiving RPC calls from the client, executing the requested functions, and sending back the result. The RPC client is responsible for connecting to ther server and making RPC calls to the server.

**model_server.py** <br/>
This file starts the RPC server and is responsible for downloading the model from WandB, loading the model, running the inference, and returning the motion direction.

**robot_client.py** <br/>
This file starts the RPC client and is responsible for capturing an image, undistroting it, saving it locally, calling the model inference function with the image filename, and then calling the robot motion function with the motion direction.
