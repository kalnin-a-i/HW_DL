from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display 
import glob
import gym
import os
import torch
import base64, io
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np


def show_video(env_name, model_name):
    '''Shows video in jupyter notebook
    
    Params
    ======
        env_name(str): name of environment
        model_name(str):model indefiter
        '''
    mp4list = glob.glob('video/*.mp4')
    if len(mp4list) > 0:
        mp4 = "video/{}/{}.mp4".format(env_name, model_name)
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")
        
def show_video_of_model(agent, env_name, model_name, max_t):
    '''Write video on disk
    
    Params
    ======
        agent(Qagent): q-learning agent
        env_name(str): name of environment
        model_name(str):model indefiter
        '''
    env = make_atari("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    if not os.path.isdir(f'video/{env_name}'):
        os.mkdir(f'video/{env_name}')
    vid = video_recorder.VideoRecorder(env, path="video/{}/{}.mp4".format(env_name, model_name))
    agent.qnetwork_local.load_state_dict(torch.load(f'{model_name}.pth'))
    state = env.reset()
    done = False
    t = 0
    while not done and t < max_t:
        state = np.array(state)
        frame = env.render(mode='rgb_array')
        vid.capture_frame()
        
        action = agent.act(state)

        state, reward, done, _ = env.step(action) 
        t += 1
    env.close()