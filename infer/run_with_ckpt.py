""" 
run mountain car continuous environment with saved ckpt
"""
import torch, os, datetime, sys, gym, pickle, time, argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.actor import Actor 
from model.critic import Critic


def scale_state(state, scaler):                 #requires input shape=(2,)
    scaled = scaler.transform([state])
    return scaled[0]                       #returns shape =(1,2)   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-actor_ckpt', type=str, help='actor ckpt path', required=True)
    parser.add_argument('-critic_ckpt', type=str, help='critic ckpt path', required=True)
    parser.add_argument('-scaler_file', type=str, help='scaler pickle file path', required=True)
    parser.add_argument('-state_size', type=int, help='state size', default=2)


    config = parser.parse_args()


    assert os.path.exists(config.actor_ckpt), 'no actor ckpt'
    assert os.path.exists(config.critic_ckpt), 'no critic ckpt'
    assert os.path.exists(config.scaler_file), 'no scaler file'

    # load scaler 
    with open(config.scaler_file, 'rb') as fd:
        scaler = pickle.load(fd)


    device = torch.device('cuda:0')

    actor = Actor(config.state_size).to(device)
    critic = Critic(config.state_size).to(device)

    actor.load_state_dict(torch.load(config.actor_ckpt))
    critic.load_state_dict(torch.load(config.critic_ckpt))


    env = gym.envs.make("MountainCarContinuous-v0")

    while True:
        state = env.reset()

        while True:

            env.render()

            state = scale_state(state, scaler)

            state = torch.FloatTensor(state).to(device)

            dist = actor(state)

            a = dist.sample()
            action = torch.tanh(a)

            action_val = action.item()

            if action_val >1:
                action_val = 1.0
            elif action_val <-1:
                action_val = -1

            next_state, reward, done, _ = env.step([action_val])

            state = next_state

            if done:
                time.sleep(1)
                break

