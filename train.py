"""
train actor/critic for continuous mountar car opengym

environment detail: https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py

"""

import torch, gym, datetime, os, json, logging, sys, numpy as np, csv, pickle, argparse
from torch.utils.tensorboard import SummaryWriter
from munch import Munch
from model.actor import Actor 
from model.critic import Critic

import sklearn
import sklearn.preprocessing

#function to normalize states
def scale_state(state, scaler):                 #requires input shape=(2,)
    scaled = scaler.transform([state])
    return scaled[0]                       #returns shape =(1,2)   

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    parser.add_argument('-suffix', type=str, help='train session save string to use as suffix', default=None)
    parser.add_argument('-actor_lr', type=float, help='actor learning rate', default=0.0001)
    parser.add_argument('-critic_lr', type =float, help='critic learning rate', default=0.0001)
    parser.add_argument('-gpu', type=int, help='if using gpu, then provide gpu number', default=None)
    parser.add_argument('-epinum', type=int, help='number of episodes to train', default=1000)
    parser.add_argument('-patient_epinum', type=int, help='episode count to wait for first success', default=10)
    parser.add_argument('-gamma', type=float, help='reward diminish gamma value', default=0.99)


    args = parser.parse_args()

    # prepare output dir
    timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    suffix = args.suffix

    outputdir = f"ckpt/{timestamp}_{suffix}"
    os.makedirs(outputdir)



    env = gym.envs.make("MountainCarContinuous-v0")


    usedconfig = Munch(vars(args))


    state_size = 2
    action_size = 1

    actor_lr = args.actor_lr
    critic_lr = args.critic_lr

    usedconfig.actor_lr = actor_lr
    usedconfig.critic_lr = critic_lr




    ### prepare state scaler

    #sample from state space for state normalization
                                        
    state_space_samples = np.array(
        [env.observation_space.sample() for x in range(10000)])
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(state_space_samples)

    # save scaler params
    scaler_params = scaler.get_params()
    savepath = os.path.join(outputdir, 'scaler_param')

    with open(savepath, 'wb') as fd:
        pickle.dump(scaler, fd)


    ## prepare tensorboard output dir
    logdir = os.path.join(outputdir, 'logs')
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)

    if args.gpu is None:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}')

    # setup actor/critic model
    actor = Actor(state_size).to(device)
    critic = Critic(state_size).to(device)



    actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)


    epi_num = args.epinum
    assert epi_num > 0, 'epi num not >0'
    max_acc_reward = None

    # number of episodes to wait until to reach first success state
    success_patient_epi_num = args.patient_epinum
    assert success_patient_epi_num > 0, 'success patient epi num not >0'
    

    restart_count = 0


    usedconfig.epi_num = epi_num
    usedconfig.success_patient_epi_num = success_patient_epi_num

    savepath = os.path.join(outputdir, 'usedconfig.json')

    with open(savepath , 'w') as fd:
        json.dump(usedconfig, fd, indent=4, ensure_ascii=False)

    # only keep on training when success state has been reached within a number of attempts.
    # if fail to do so, then reset actor/critics weights and try again to reach success state
    success_reached=False

    epi_index = 0
    patient_count = 0
    max_step = 200

    for _ in range(epi_num):

        if patient_count > success_patient_epi_num and success_reached is False:
            print(f"### failed to reach success with in patient rounds: {success_patient_epi_num}. restart_count={restart_count}")

            restart_count+=1

            # reinit weights in actor and critic
            actor.initialize()
            critic.initialize()

            patient_count =0
            

        # for each episode, keep track of rewards, states, logprobs, value of all steps
        state = env.reset()
        state = scale_state(state, scaler)

        state = torch.FloatTensor(state).to(device)


        step_count= 0

        rewards = []
        log_probs = []
        curr_values = []
        acc_reward = 0 # for tracking overall performance
        entropy = 0 # not used in loss. tracking for viewing

        while True:

            step_count += 1

            dist = actor(state)

            entropy += dist.entropy().mean()

            a = dist.sample()
            action = torch.tanh(a)

            
            log_prob = dist.log_prob(a).unsqueeze(0)

            log_probs.append(log_prob)

            action_val = action.item()

            # clip action value
            if action_val >1:
                action_val = 1.0
            elif action_val <-1:
                action_val = -1

            next_state, reward, done, _ = env.step([action_val])

            if step_count > max_step:
                # if step exceed max step. abort episode. give negative reward manually(optional.)
                # the mountain car cont. environment doesn't seem to finish itself after 200 steps even 
                # though the docstrings say that it terminates after 200 steps.
                reward = -100
                done = True
        
            next_state = scale_state(next_state, scaler)

            acc_reward += reward

            rewards.append(torch.FloatTensor([reward]).to(device))

            next_state = torch.FloatTensor(next_state).to(device)

            curr_value = critic(state)

            curr_values.append(curr_value)

            if reward >0:
                success_reached = True
                
            state = next_state

            if done:
                break


        # get advantages
        advantages = []
        returns = []
        r = 0
        gamma = args.gamma 
        for reward in reversed(rewards):
            r = reward + gamma * r
            returns.insert(0, r)

        returns = torch.FloatTensor(returns).to(device).detach()
        curr_values = torch.FloatTensor(curr_values).to(device)
        advantages = returns - curr_values



        log_probs = torch.cat(log_probs)

        actor_loss = - (log_probs * advantages).mean()
        critic_loss = advantages.pow(2).mean()

        actor_opt.zero_grad()
        critic_opt.zero_grad()

        (actor_loss + critic_loss).backward()

        actor_opt.step()
        critic_opt.step()

        print(f"epi:{epi_index}, steps={step_count}, entropy={entropy.item()}, acc_reward={acc_reward} actor_loss: {actor_loss.item()}, critic_loss: {critic_loss.item()}")

        # save progress to tensorboard writer
        writer.add_scalar('train/actor_loss', actor_loss.item(), epi_index)
        writer.add_scalar('train/critic_loss', critic_loss.item(), epi_index)
        writer.add_scalar('train/steps', step_count, epi_index)
        writer.add_scalar('train/acc_rewards', acc_reward, epi_index)
        writer.add_scalar('train/entropy', entropy.item(), epi_index)
        writer.flush()

        # save model when better performance is detected
        if max_acc_reward is None or acc_reward >= max_acc_reward:

            max_acc_reward = acc_reward

            savedir = os.path.join(outputdir, f'epi_{epi_index}_acc_reward={acc_reward:.2f}')
            os.makedirs(savedir)

            savepath = os.path.join(savedir, 'actor.pt')
            torch.save(actor.state_dict(), savepath)

            savepath = os.path.join(savedir, 'critic.pt')
            torch.save(critic.state_dict(), savepath)

        
        epi_index+=1

        if not success_reached:
            patient_count +=1
