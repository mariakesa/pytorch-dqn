import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from gym.wrappers import Monitor
import datetime
import os
import torch
import numpy as np

from model import Model
from environment import Environment
from utils import Memory, EpsilonScheduler, make_log_dir, save_gif

from EnsemblePursuit.EnsemblePursuit import EnsemblePursuit

from rastermap import Rastermap
from scipy import stats

from sklearn.decomposition import PCA, NMF

def running_average(X, nbin = 100):
    Y = np.cumsum(X, axis=0)
    Y = Y[nbin:, :] - Y[:-nbin, :]
    return Y

def rm(ts):
    model = Rastermap(n_components=1, n_X=100).fit(ts)
    isort = np.argsort(model.embedding[:,0])

    # sort by embedding and smooth over neurons
    Sfilt = running_average(ts[isort, :500], 50)
    Sfilt = stats.zscore(Sfilt, axis=1)
    plt.figure(figsize=(16,12))
    plt.imshow(Sfilt, vmin = -0.5, vmax=3, aspect='auto', cmap='gray_r')
    plt.xlabel('time points')
    plt.ylabel('sorted neurons')
    plt.show()


def test(q_func,DEVICE,NUM_TEST,env,save=False):
    print("[TESTING]")
    total_reward = 0
    unclipped_reward = 0
    im_arr=[]
    for i in range(NUM_TEST):
        if i == 0 and save:
            frames = []

        env.reset(eval=True) # performs random actions to start
        state, _, done, _ = env.step(env.action_space.sample())
        frame = 0

        while not done:
            if i == 0 and save:
                frames.append(state[0,0])
            im_arr.append(state[0,0].detach().numpy().flatten())
            # env.render()
            q_values = q_func(state.to(DEVICE))
            if np.random.random() > 0.01: # small epsilon-greedy, sometimes 0.05
                action = torch.argmax(q_values, dim=1).item()
            else:
                action = env.action_space.sample()

            lives = env.ale.lives()
            next_state, reward, done, info = env.step(action)
            if env.ale.lives() != lives: # lost life
                pass
                # plt.imshow(next_state[0,0])
                # plt.savefig(f"frame-{frame}.png")
                # print("LOST LIFE")

            unclipped_reward += info['unclipped_reward']
            total_reward += reward
            state = next_state
            frame += 1
            # print(f"[TESTING {frame}] Action: {action}, Q-Values: {np.array(q_values.cpu().detach())}, Reward: {reward}, Total Reward: {total_reward}, Terminal: {done}")
            # plt.imshow(state[0,0])
            # plt.savefig("frame-{}.png".format(frame))

        if i == 0 and save:
            frames.append(state[0,0])
            save_gif(frames, "{}.gif".format(os.path.join(video_dir, str(scheduler.step_count()))))

    total_reward /= NUM_TEST
    unclipped_reward /= NUM_TEST
    print(f"[TESTING] Total Reward: {total_reward}, Unclipped Reward: {unclipped_reward}")


    return total_reward, q_func.recorder.data, np.array(im_arr)

def run_net(game):
    if game=='space invaders':
        weights_str='space-invaders'

    weights_path='/home/maria/Documents/pytorch-dqn/weights/'+weights_str+'/good.pt'

    MEM_SIZE = int(1e6) # this is either 250k or 1 million in the paper (size of replay memory)
    EPISODES = int(1e5) # total training episodes
    BATCH_SIZE = 32 # minibatch update size
    GAMMA = 0.99 # discount factor
    STORAGE_DEVICES = ['cpu'] # list of devices to use for episode storage (need about 10GB for 1 million memories)
    DEVICE = 'cpu' # list of devices for computation
    UPDATE_FREQ = 4 # perform minibatch update once every UPDATE_FREQ
    TARGET_UPDATE_EVERY = 10000 # in units of minibatch updates
    INIT_MEMORY_SIZE = 200000 # initial size of memory before minibatch updates begin

    TEST_EVERY = 1000 # (episodes)
    PLOT_EVERY = 10 # (episodes)
    SAVE_EVERY = 1000 # (episodes)
    EXPERIMENT_DIR = "experiments"
    NUM_TEST = 1
    GAME = game

    env = Environment(game=GAME)
    #mem = Memory(MEM_SIZE, storage_devices=STORAGE_DEVICES, target_device=DEVICE)

    q_func = Model(env.action_space.n).to(DEVICE)
    q_func.load_state_dict(torch.load(weights_path,map_location='cpu'))

    target_q_func = Model(env.action_space.n).to(DEVICE)
    target_q_func.load_state_dict(q_func.state_dict())

    _, activations,im_arr=test(q_func,DEVICE,NUM_TEST,env)
    return activations, q_func,im_arr

def convert_single_layer(ind,activations):
    shp=activations[ind][0].shape
    print(shp)
    ts=np.zeros((np.prod(shp),len(activations[0])))
    for j in range(0,len(activations[0])):
        #print(activations[ind][j].detach().numpy().flatten().shape)
        ts[:,j]=activations[ind][j].detach().numpy().flatten()
    return ts
def all_neurons_ts(inds,activations):
    ts=np.zeros((1,len(activations[0])))
    for j in inds:
        ts_=convert_single_layer(j,activations)
        ts=np.vstack((ts,ts_))
    ts=ts[1:,:]
    return ts

def compute_rfield(imgs,V):
    imgs=imgs.T
    lam=10
    npix=84*84
    B0 = np.linalg.solve((imgs @ imgs.T + lam * np.eye(npix)),  (imgs @ V)).reshape(84,84,15)
    return B0

def receptive_fields_of_ind_layers(t_arr,imgs):
    neurons=t_arr.T
    ep=EnsemblePursuit(n_components=15,n_kmeans=15,lam=0.1)
    ep.fit(neurons)
    V=ep.components_
    U=ep.weights.flatten()
    print(V.shape)
    B0=compute_rfield(imgs,V)
    for j in range(0,15):
        n_neurons=np.nonzero(ep.weights[:,j].flatten())[0].shape[0]
        plt.imshow(B0[:,:,j],cmap='bwr')
        plt.title('Ensemble '+str(j)+', n_neurons='+str(n_neurons))
        plt.show()

def pca_receptive_fields_of_ind_layers(t_arr,imgs):
    neurons=t_arr.T
    pcs=PCA(n_components=15).fit_transform(t_arr.T)
    B0=compute_rfield(imgs,pcs)
    for j in range(0,15):
        plt.imshow(B0[:,:,j],cmap='bwr')
        plt.title('PC '+str(j))
        plt.show()

def nmf_receptive_fields_of_ind_layers(t_arr,imgs):
    neurons=t_arr.T
    comps=NMF(n_components=15).fit_transform(t_arr.T)
    B0=compute_rfield(imgs,comps)
    for j in range(0,15):
        plt.imshow(B0[:,:,j],cmap='bwr')
        plt.title('Component '+str(j))
        plt.show()
