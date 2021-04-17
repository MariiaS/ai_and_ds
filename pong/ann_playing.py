"""
This script makes the trained agent play Pong against the computer.
After training the agent, you can just run this file to see the agent play. 

Note: The game will often just freeze after 21 points are scored by either player.
You can just kill the terminal you ran this file from to exit it.
"""
import gym
from imageio import imwrite
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model (saved in ann_training.py)
model = keras.models.load_model('trained_agent')

# The moves our agent can make
move_ids = [0, 2, 3]


def write_next_state(game_env, action, obs_now):
    """
    This function takes in the game environment, the action to be taken, and the current state of the game
    It makes the action, takes the new state of the game, and stores the difference in an image, 
    that will be fed to the model to decide the next move.
    """
    obs_new, _, _, _ = game_env.step(action)
    present = obs_now[34:194:4,12:148:2,1] #frame before action
    futur = obs_new[34:194:4,12:148:2,1] #frame after action was taken
    data = present - futur
    imwrite("current_state.png", data)
    return obs_new # return state of game after action was taken
    
env = gym.make('Pong-v4')
env.reset()
env.render()
# get initial values for the state of the game and the image data to give our model
obs, _, _, _ = env.step(env.action_space.sample())
obs = write_next_state(env, env.action_space.sample(), obs)

while True:
    env.render()
    # loading the current state to give it to our model
    img = keras.preprocessing.image.load_img("current_state.png", target_size=(40, 68), color_mode="grayscale")
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) #adding the batch dimension because we have to
    # the scores of each moves in the current state according to our model
    pred = model.predict(img_array)[0]
    #print(pred) #if you want to see the three scores
    action = move_ids[np.argmax(pred)] #the actual action to be taken, the one of max score

    #make the move, update the current_state.png
    obs = write_next_state(env, action, obs)


env.close()