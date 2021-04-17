"""
The purpose of this script is to create our database.

HOW TO USE:
1. In the directory where this script is located, create a folder called "data" with three subfolders "0", "2", and "3"
2. From the directory where this script is located, run the command: python data_creation.py
This will launch a game of Pong. Use the 'z' and 's' buttons to play. 
Every frame gets saved in the subfolder corresponding to the action taken on that frame.
You can end the game at any time by closing the game window.

Note: if you stopped recording frames but want to create more data without erasing the existing data,
modify the starting value of the count attribute in the Saver class
"""

import numpy as np
import gym
from gym.utils.play import play
from imageio import imwrite


class Saver():
    """Utilitary class to make the saving process easier"""
    
    def __init__(self, directory="./data"):
        """resets the state counter to the default 'start of game' value."""
        self.dir = directory
        self.count = -22 #we don't save the first frames where the game hasn't started yet
    
    def save(self, data, label):
        """Saves the data to the folder corresponding to the data's label"""
        if self.count >= 0:
            imwrite(self.dir + "/" + str(label) + "/state_" + str(self.count) + ".png", data)
        self.count += 1
        print(f"Saved frames: {self.count}")


s = Saver()
def get_data(obs_t, obs_tp1, action, rew, done, info):
    # we want to include temporal information, 
    # so we do the difference of the frame pre-action with the frame post-action
    # (we also subsample, crop, and keep only one color channel)
    present = obs_t[34:194:4,12:148:2,1]
    futur = obs_tp1[34:194:4,12:148:2,1]
    data = present - futur
    s.save(data, action)

if __name__ == '__main__':
    print("Hit ESC key to quit.")
    env = gym.make('Pong-v4')
    env.reset()
    keys_to_action = {(ord('z'),): 2, (ord('s'),): 3}
    play(env, zoom=3, fps=13, callback=get_data, keys_to_action=keys_to_action)
    env.close()

