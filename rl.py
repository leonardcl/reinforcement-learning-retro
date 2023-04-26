import retro
import gym
import numpy as np
import time
import os
import matplotlib.pyplot as plt
# from IPython.display import clear_output
import cv2
import math
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn as nn
import csv
import codecs
from collections import deque
import pathlib
from resnet import ft_net
from model import *
import argparse


def str2bool(value, raise_exc=False):
    _true_set = {'yes', 'true', 't', 'y', '1'}
    _false_set = {'no', 'false', 'f', 'n', '0'}
    if isinstance(value, str):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

    if raise_exc:
        raise ValueError('Expected "%s"' % '", "'.join(_true_set | _false_set))
    return None

def preprocess_frame(screen, exclude, output):
    """Preprocess Image.
        
        Params
        ======
            screen (array): RGB Image
            exclude (tuple): Section to be croped (UP, RIGHT, DOWN, LEFT)
            output (int): Size of output image
        """
    # TConver image to gray scale
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    
    #Crop screen[Up: Down, Left: right] 
    screen = screen[exclude[0]:exclude[2], exclude[3]:exclude[1]]
    
    # Convert to float, and normalized
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    
    # Resize image to 84 * 84
    # screen = cv2.resize(screen, (output, output), interpolation = cv2.INTER_AREA)
    screen = cv2.resize(screen, (84, 84), interpolation = cv2.INTER_AREA)
    return screen


def stack_frame(stacked_frames, frame, is_new):
    """Stacking Frames.
        
        Params
        ======
            stacked_frames (array): Four Channel Stacked Frame
            frame: Preprocessed Frame to be added
            is_new: Is the state First
        """
    if is_new:
        stacked_frames = np.stack(arrays=[frame, frame, frame, frame])
        stacked_frames = stacked_frames
    else:
        stacked_frames[0] = stacked_frames[1]
        stacked_frames[1] = stacked_frames[2]
        stacked_frames[2] = stacked_frames[3]
        stacked_frames[3] = frame
    
    return stacked_frames

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (30, -1, -1, 1), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

def data_write_csv(file_name, datas): # file_name is the path to write the CSV file, datas is the list of data to be written
    file_csv = codecs.open(file_name,'w+','utf-8') # append
    writer = csv.writer(file_csv)
    for data in datas:
        writer.writerow(data)

    
    

def train(n_episodes, alg):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    score_max = 0
    coins_max = 0
    score_history = [[]]
    reward_history = [[]]
    for i_episode in range(start_epoch + 1, n_episodes+1):
        state = stack_frames(None, env.reset(), True)
        score = 0
        xscrollLo = 0
        reward_e = 0
        xscrollLo_prev = 0
        xscrollHi = 0
        xscrollHi_prev = 0
        score_max = 0
        coins_max = 0   
        health = 0
        health_prev = 2

        eps = epsilon_by_epsiode(i_episode)

        # Punish the agent for not moving forward
        # steps_stuck = 0
        timestamp = 0
        print(i_episode)
        counter = 0
        while timestamp < 10000:
            env.render()
            # print(timestamp, end="")
            # action = agent.act(state, eps)
            if alg == "DQN":
                action = agent.act(state, eps)
            else:
                action, log_prob, value = agent.act(state)
            next_state, reward, done, info = env.step(possible_actions[action])
                
            
            timestamp += 1

            score += reward
            # Punish the agent for not gain score.
            # reward = 0
            # xscrollLo = info['xscrollLo']
            # xscrollHi = info['xscrollHi']
            # coins = info['coins']
            # health = info['lives']
            # lives = info['lives']
            # time = info['time']
            # scrolling = info['scrolling']
            
            #####################################
            # if coins > coins_max:
            #     reward += 1000
            #     coins_max = coins
                
            # if score > score_max:
            #     reward += 1000
            #     score_max = score
                
            # if xscrollLo > xscrollLo_prev:
            #     reward += 50
            #     xscrollLo_prev = xscrollLo
            #     counter = 0
            # else:
            #     counter += 1
            #     reward -= 0.1 * (counter//10)
                
            # if xscrollHi > xscrollHi_prev:
            #     reward += 2000
            #     xscrollHi_prev = xscrollHi
            #     xscrollLo_prev = 0
            #####################################
                
            # if 2 > health and (info['time'] > 0 and info['time'] != 400):
            #     # reward -= 10000000
            #     done = True
                
            # 20230423_1048 --> reward - 5 * (2 - health) 
            # 20230423_1549 --> reward scrollLo += 25 reward score_max += 1000 reward coins_max += 1000
            # 20230423_2033 --> No additional reward function, 3000 episode
            # 20230424_2210 --> Resnet, no additional reward function
            # 20230525_0647 --> DQN, Own rewward function
            # 20230525_1152 --> DQN, additional reward function
            # 20230526_1323 --> A2C, additional reward function
                
            # if (prev_state['score'] == info['score']):
            #     steps_stuck += 1
            # else:
            #     steps_stuck = 0
            # if prev_state['lives'] > info['lives']:
            #     reward -= 1000
    
            # if (steps_stuck > 40):
            #     reward -= 10
            reward_e += reward
            
            next_state = stack_frames(state, next_state, False)
            # agent.step(state, action, reward, next_state, done)
            # agent.step(state, log_prob, entropy, reward, done, next_state)
            if alg == "DQN":
                agent.step(state, action, reward, next_state, done)
                # action, log_prob, entropy = agent.act(state)
            else:
                agent.step(state, action, value, log_prob, reward, done, next_state)
            
            state = next_state
            if done:
                break
            
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        rewards_window.append(reward_e)       # save most recent reward
        rewards.append(reward_e) 
        
        # clear_output(True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')        
        fig.savefig("mario_score.png")
        plt.close(fig)
        
        agent.save_networks(i_episode, n_episodes)
        
        score_data = [i_episode, score, np.mean(scores_window), eps]
        score_history.append(score_data)
        data_write_csv("./csv/mario_score_history.csv",  score_history)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(rewards)), rewards)
        plt.ylabel('Reward')
        plt.xlabel('Episode #')        
        fig.savefig("mario_reward.png")
        plt.close(fig)
        
        # agent.save_networks(i_episode, n_episodes)
        
        reward_data = [i_episode, reward_e, np.mean(rewards_window), eps]
        reward_history.append(reward_data)
        data_write_csv("./csv/mario_reward_history.csv",  reward_history)

        print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, score, np.mean(scores_window), eps))
        # print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window), eps), end="")
    
    return scores

def test_agent(n_timesteps, epsilon, n_episodes, alg):
    print("Testing Starts")
    # env.viewer = None
    start_epoch = 0
    success_count = 0

    for i_episode in range(start_epoch + 1, n_episodes + 1):
        state = stack_frames(None, env.reset(), True)
        score = 0
        timestamp = 0
        while timestamp < n_timesteps:
            env.render()
            if alg == "DQN":
                action = agent.act_test(state, epsilon)
            else:
                action, log_prob, value = agent.act_test(state)
                
            next_state, reward, done, info = env.step(possible_actions[action])

            timestamp += 1
            # print(timestamp)
            score += reward
            state = stack_frames(state, next_state, False)
            if done:
                break

        acc_score = (score/i_episode)

        print("\rEpisode: {}\Score average: {:.2f}".format(i_episode, acc_score), end="")



parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train_mode", help="Whether to train or test the model", default=False)
parser.add_argument("-al", "--alg", help="What algorithm to use, DQN or A2C", default="DQN")
parser.add_argument("-lm", "--load_model", help="Whether to Load Pre-Trained Policy & Target Network or not", default=False)
parser.add_argument("-gm", "--game", help="What game do you want to run, Super Mario Bros or Felix The Cat", default="SMB")
parser.add_argument("-mn", "--model_name", help="Name of the policy & target network, separated by comma (,). No spaces",
                    default="mario_policy_net,mario_target_net")

args = parser.parse_args()
train_mode = str2bool(args.train_mode)
alg = args.alg
game = args.game
LOAD_MODEL = str2bool(args.load_model)

if game == "SMB":
    env = retro.make("SuperMarioBros-Nes", "Level1-1")
    possible_actions = {
            # No Operation
            0: [0, 0, 0, 0, 0, 0, 0, 0, 0],
            # # Left
            # 1: [0, 0, 0, 0, 0, 0, 1, 0, 0],
            # Right
            1: [0, 0, 0, 0, 0, 0, 0, 1, 0],
            # # Left, A
            # 3: [0, 0, 0, 0, 0, 0, 1, 0, 1],
            # Right, A
            2: [0, 0, 0, 0, 0, 0, 0, 1, 1],
            # A
            3: [0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
else:
    env = retro.make("FelixTheCat-Nes")
    possible_actions = {
                # No Operation
                0: [0, 0, 0, 0, 0, 0, 0, 0, 0],
                # Left
                1: [0, 0, 0, 0, 0, 0, 1, 0, 0],
                # Right
                2: [0, 0, 0, 0, 0, 0, 0, 1, 0],
                # Left, B
                3: [1, 0, 0, 0, 0, 0, 1, 0, 0],
                # Right, B
                4: [1, 0, 0, 0, 0, 0, 0, 1, 0],
                # B
                5: [1, 0, 0, 0, 0, 0, 0, 0, 0],
                # Left, A
                6: [0, 0, 0, 0, 0, 0, 1, 0, 1],
                # Right, A
                7: [0, 0, 0, 0, 0, 0, 0, 1, 1],
                # A
                8: [0, 0, 0, 0, 0, 0, 0, 0, 1]
            }

MODEL_NAME = args.model_name.split(",")
    
INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = len(possible_actions)
SEED = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

if alg == "DQN":
    # DQN
    GAMMA = 0.99           # discount factor
    BUFFER_SIZE = 100000   # replay buffer size
    BATCH_SIZE = 32        # Update batch size
    LR = 0.0001            # learning rate 
    TAU = 1e-3             # for soft update of target parameters
    UPDATE_EVERY = 50      # how often to update the network
    UPDATE_TARGET = 2000   # After which thershold replay to be started 
    EPS_START = 0.99       # starting value of epsilon
    EPS_END = 0.09        # Ending value of epsilon
    EPS_DECAY = 150         # Rate by which epsilon to be decayed
    if game == "SMB":
        agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn, LOAD_MODEL, MODEL_NAME)
    else:
        agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn_Sonic, LOAD_MODEL, MODEL_NAME)

else:
    # PPO
    GAMMA = 0.99           # discount factor
    ALPHA= 0.0001          # Actor learning rate
    BETA = 0.0001          # Critic learning rate
    TAU = 0.95
    BATCH_SIZE = 32
    PPO_EPOCH = 5
    CLIP_PARAM = 0.2
    UPDATE_EVERY = 1000     # how often to update the network 
    agent = PPOAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, GAMMA, ALPHA, BETA, TAU, UPDATE_EVERY, BATCH_SIZE, PPO_EPOCH, CLIP_PARAM, ActorCnn, CriticCnn, LOAD_MODEL, MODEL_NAME)

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)
rewards = []
rewards_window = deque(maxlen=20)
epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)

if train_mode:
    scores = train(1000, alg)
else:
    test_agent(10000, 0.09, 10, alg)