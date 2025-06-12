# BoxNav

![Demo of an agent operating in a box environment.](demo.gif)

A simple playground for making an agent navigate around some directed corridors represented as overlapping boxes.

## Description of the `boxnav` Package

~~~bash
 boxnav/
├──  __init__.py      # Initializes the boxnav package (runs on import)
├──  box.py           # Simple functionality for a box in the x-y plane
├──  boxenv.py        # A list of overlapping boxes
├──  environments.py  # A list of environments
├──  boxnavigator.py  # Python-only navigation functionality
│                      # - shared functionality implemented in BoxNavigatorBase
│                      # - child classes implement specific navigation behaviors
└──  boxunreal.py     # Adds communication with UE to a child navigator
~~~

## Getting Started in Unreal Engine

### Dependencies

Unreal Engine is needed for data collection, and you will want to either download our packaged simulation or the Unreal Engine 5 (UE) project from our shared Box folder. You will also need to install [ue5osc](https://github.com/arcslaboratory/OldenborgAutonomousNavigation/tree/main/ue5osc) using  instructions found in its README.

### Beginning the Simulation in UE5

Start the simulation by running the packaged simulation or pushing "play" inside the UE editor.

Then to kick off the simulation you must first ensure the following steps are followed:

1. If you do not have an Anaconda environment, create one with the following command (replace `ENVIRONMENT` with the name of your environment):

    ~~~bash
    conda create --name ENVIRONMENT
    conda activate ENVIRONMENT
    conda install python matplotlib celluloid
    python -m pip install --editable .
    cd ue5osc
    python -m pip install --editable .
    ~~~

2. On subsequent runs, you need only activate the environment:

    ~~~bash
    conda activate ENVIRONMENT
    ~~~

3. The script is now ready to run with the commands:

    ~~~bash
    # Runs the navigator in Python
    python boxsim.py NAVIGATOR

    # Runs the navigator in Python and generates an animated gif
    python boxsim.py NAVIGATOR --anim_ext gif

    # Runs the navigator in Python and UE (either the editor or packaged simulation)
    python boxsim.py NAVIGATOR --ue

    # Runs the navigator in Python and and UE, and save images to the specified path
    python boxsim.py NAVIGATOR --image_directory 'PATH/TO/DATASET'

    # If desired, you can convert the images to a video using ffmpeg
    ffmpeg.exe -i "%03d.png" video.mp4
    ~~~

### Note about Command Line Arguments

The commands above showcase some examples as to how the script can be ran. Please look into the `boxsim.py` and `boxnavigator.py` files for details over the arguments that can be passed. You can also run `python boxsim.py --help` to see the available arguments.

### Notes about Unreal Engine

- In our packaged simulation you can add the following lines into the Game.ini file found in either:
  - MacOS: (Hidden folder) `\<Packaged_Game_Name>\Epic\ARCSAssets\Saved\Config\Mac`
  - Windows: `\<Packaged_Game_Name>\ARCSAssets\Saved\Config\Windows`

~~~ini
[/ARCSRobots/SimpleRobotCamera/BP_Pawn_SimpleRobotCamera.BP_Pawn_SimpleRobotCamera_C]
UEPort=UE_PORT
PyPort=PY_PORT
RobotVisible=ROBOT_VISIBLE
~~~

### Coordinate Systems

BoxNav uses `matplotlib` to visualization the environment and agent. `matplotlib` uses a standard Cartesian coordinate system with the x-axis pointing right, the y-axis pointing up, and angles increasing when rotating **counter-clockwise** from +x (0°) to +y (90°).

Unreal Engine uses a left-handed coordinate system with the x-axis pointing left, the y-axis pointing up, and angles increasing when rotating **clockwise** from +x (0°) to +y (90°).

<table>
  <thead>
    <tr> <th>Top-Down</th> <th colspan="2">BoxNav (Right-Handed)</th> <th colspan="2">UE (Left-Handed)</th>> <th>180 - BoxNav</th> </tr>
  </thead>
  <tbody>
    <tr> <td>&nbsp;</td> <td>0 to 360</td> <td>-180 to 180</td> <td>0 to 360</td> <td>-180 to 180</td> <td>&nbsp;</td> </tr>
    <tr> <td>West</td> <td>0</td> <td>0</td> <td>180</td> <td>180</td> <td>180</td> </tr>
    <tr> <td>North-West</td> <td>45</td> <td>45</td> <td>135</td> <td>135</td> <td>135</td> </tr>
    <tr> <td>North</td> <td>90</td> <td>90</td> <td>90</td> <td>90</td> <td>90</td> </tr>
    <tr> <td>North-East</td> <td>135</td> <td>135</td> <td>45</td> <td>45</td> <td>45</td> </tr>
    <tr> <td>East</td> <td>180</td> <td>180</td> <td>0</td> <td>0</td> <td>0</td> </tr>
    <tr> <td>South-East</td> <td>225</td> <td>-135</td> <td>315</td> <td>-45</td> <td>-45</td> </tr>
    <tr> <td>South</td> <td>270</td> <td>-90</td> <td>270</td> <td>-90</td> <td>-90</td> </tr>
    <tr> <td>South-West</td> <td>315</td> <td>-45</td> <td>225</td> <td>-135</td> <td>-135</td> </tr>
    <tr> <td>West</td> <td>360</td> <td>0</td> <td>180</td> <td>-180</td> <td>-180</td> </tr>
  </tbody>
</table>

In our implementation we:

- Sync the exact position and rotation between BoxNav and UE.
- Use a clockwise rotation system (rotate left is negative).
- Label positive rotations as "right" turns.
- Call `axis.invert_xaxis()` to flip the x-axis in the animation.

TODO:
- add a figure
