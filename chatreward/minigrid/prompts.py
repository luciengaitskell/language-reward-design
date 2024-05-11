prompt1 = """You are an assistant aiding with subgoal generataion for reinforcement learning problems. Specifically, you will
be given an example environment picture and a textual goal description, and you are to output language subgoals that the agent
should achieve in order to efficiently and successfully achieve the main goal.

The goal description is:

“get the {lockedroom_color} key from the {keyroom_color} room, unlock the {door_color} door and go to the goal”

where you can use {lockedroom_color}, {keyroom_color}, and {door_color} as variables in the subgoals and the goal is a light green square.
The variables can be the values “red”, “green”, “blue”, “purple”, “yellow” or “grey”.

These subgoals should be with respect to the image itself: they should specify specific observations that
show that the agent is on track. They should also be singular; they should output exactly one subtask instead of a group of multiple steps.
Output a list of ONLY these text subgoals in the following format (without any introduction text):

- [subgoal 1]
- [subgoal 2]
- ...

where [subgoal i] is replaced by the ith subgoal. subgoals Do not create directional subgoals but rather strategic
subgoals that do not hard code the direction but instead tell the agent which states are more beneficial. The subgoals should be able to be completed
by the agent moving, picking up a key, or using a key.

You should enough subgoals to be descriptive but not redundant. Use the included example image to be specific.

Remember that the included image is an example of the environment but the door, key, and goal locations may differ so use it for context but do not
hardcode the goals with respect to this specific image.
"""

prompt2 = """
You are an assistant tasked with turning language subgoals into machine readable code. You will be given text subgoals, and you must translate
these subgoals into code that takes in an observation of the format

{'direction': Discrete(4), 'image': np.ndarray, 'mission': str}

'direction' is a number with the corresponding direction:
    3: Up
    2: Left
    1: Down
    0: Right

'image' is of shape (width x height x 3), and the final dimension corresponds to a 3D tuple of (OBJECT_IDX, COLOR_IDX, STATE). Here, 

OBJECT_IDX is a number with the corresponding object type: 
    0: "unseen"
    1: "empty"
    2: "wall"
    3: "floor"
    4: "door"
    5: "key"
    
COLOR_IDX is a number with the corresponding color:
    0: "red"
    1: "green"
    2: "blue"
    3: "purple"
    4: "yellow"
    5: "grey"

STATE is a number with the corresponding state if the object is a door: 
    0: "open"
    1: "closed"
    2: "locked"

The agent is centered at the bottom of the image.

'mission' is of the form “get the {lockedroom_color} key from the {keyroom_color} room, unlock the {door_color} door and go to the goal”

The text subgoals will include these variables {lockedroom_color}, {keyroom_color}, and {door_color}, which are integers corresponding to the above encoding for COLOR_IDX.

As part of your reward function, you can use these three variables {lockedroom_color}, {keyroom_color}, and {door_color}
to determine your specific reward function for the subgoal. You can also use the action which comes as an integer 0 through 6, labeled as follows:

0: turn left, 1: turn right, 2: move forward, 3: pick up an object, 4: DON'T USE, 5: toggle/activate an object, 6: DON'T USE

The reward function must have the signature: reward(observation, action, lockedroom_color, keyroom_color, door_color). 
 
Output the subgoal as a  python functions that takes in the above parameters and returns the reward that prioritizes the specific subgoal. The only external library you can use is numpy.
This reward function should be dense; it should make the agent want to move closer to the specific subgoal. For example, you can use a distance function to achieve this for certain subgoals. You can also
add reward components to ensure the prerequisites for the current subgoal are achieved if you think that would help.
Have the maximum reward of the function be 1 (where the goal is obtained) and the minimum be 0. DO NOT use float('inf') or float('-inf') in your function.
The code should be the only thing you output, all in one python function without sub functions."""
