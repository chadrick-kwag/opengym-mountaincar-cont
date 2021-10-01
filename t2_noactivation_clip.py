"""
https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py

"""

import torch, gym, datetime, os, json
from torch.utils.tensorboard import SummaryWriter
from munch import Munch


env = gym.envs.make("MountainCarContinuous-v0")

print(env.observation_space.shape)

print(env.action_space.shape)

usedconfig = Munch()


state_size = 2
action_size = 1

actor_lr = 0.00002
critic_lr = 0.001

usedconfig.actor_lr = actor_lr
usedconfig.critic_lr = critic_lr


class Actor(torch.nn.Module):

    def __init__(self, state_size):

        super().__init__()

        self.state_size = state_size


        self.lin1 = torch.nn.Linear(self.state_size, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.lin3 = torch.nn.Linear(128, 2)

    def forward(self, state):

        y = torch.relu(self.lin1(state))
        y = torch.relu(self.lin2(y))
        y = self.lin3(y)

        mu = y[0]
        sigma = y[1]

        sigma = torch.exp(sigma)

        dist = torch.distributions.Normal(mu, sigma)

        return dist


class Critic(torch.nn.Module):

    def __init__(self, state_size):

        super().__init__()

        self.lin1 = torch.nn.Linear(state_size, 256)
        self.lin2 = torch.nn.Linear(256, 128)
        self.lin3 = torch.nn.Linear(128, 1)

    def forward(self, state):

        y = torch.relu(self.lin1(state))
        y = torch.relu(self.lin2(y))
        y = self.lin3(y)

        return y



timestamp=datetime.datetime.now().strftime("%y%m%d_%H%M%S")

suffix = "split_actor_and_critic_lr"

outputdir = f"ckpt/t1/{timestamp}_{suffix}"
os.makedirs(outputdir)





logdir = os.path.join(outputdir, 'logs')
os.makedirs(logdir)

writer = SummaryWriter(logdir)


device = torch.device('cuda:0')

actor = Actor(state_size).to(device)
critic = Critic(state_size).to(device)



actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)


epi_num = 10000
max_acc_reward = None

success_patient_epi_num = 20


# save run info
usedconfig.epi_num = epi_num
usedconfig.success_patient_epi_num = success_patient_epi_num

savepath = os.path.join(outputdir, 'usedconfig.json')

with open(savepath , 'w') as fd:
    json.dump(usedconfig, fd, indent=4, ensure_ascii=False)


done_reached=False

for epi_index in range(epi_num):

    if epi_index > success_patient_epi_num and done_reached is False:
        print(f"### failed to reach success with in patient rounds: {success_patient_epi_num}")
        break

    state = env.reset()

    state = torch.FloatTensor(state).to(device)


    step_count= 0

    rewards = []
    log_probs = []
    curr_values = []
    acc_reward = 0
    entropy = 0

    while True:

        step_count += 1

        env.render()

        dist = actor(state)

        entropy += dist.entropy().mean()

        a = dist.sample()
        action = torch.tanh(a)
        log_prob = dist.log_prob(a).unsqueeze(0)
        # print(log_prob)

        log_probs.append(log_prob)

        # print(action)

        next_state, reward, done, _ = env.step([action.item()])

        acc_reward += reward

        rewards.append(torch.FloatTensor([reward]).to(device))

        next_state = torch.FloatTensor(next_state).to(device)

        curr_value = critic(state)

        curr_values.append(curr_value)
        # next_value = critic(next_state)



        if done:
            done_reached = True
            # print(f'reward: {reward}')
            break

    next_value = critic(next_state)

    # get advantages
    advantages = []
    returns = []
    r = 0
    gamma = 0.99
    for reward in reversed(rewards):
        r = reward + gamma * r
        returns.insert(0, r)

    returns = torch.FloatTensor(returns).to(device).detach()
    curr_values = torch.FloatTensor(curr_values).to(device)
    advantages = returns - curr_values


    # advantage = return - currvalue
    # for _return, value in zip(returns, curr_values):
    #     advantage = _return - value
    #     advantages.append(advantage)

    # advantages = torch.cat(advantages)
    log_probs = torch.cat(log_probs)

    actor_loss = - (log_probs * advantages).mean()
    critic_loss = advantages.pow(2).mean()

    actor_opt.zero_grad()
    critic_opt.zero_grad()

    (actor_loss + critic_loss).backward()

    actor_opt.step()
    critic_opt.step()

    print(f"epi:{epi_index}, steps={step_count}, entropy={entropy.item()}, acc_reward={acc_reward} actor_loss: {actor_loss.item()}, critic_loss: {critic_loss.item()}")


    writer.add_scalar('train/actor_loss', actor_loss.item(), epi_index)
    writer.add_scalar('train/critic_loss', critic_loss.item(), epi_index)
    writer.add_scalar('train/steps', step_count, epi_index)
    writer.add_scalar('train/acc_rewards', acc_reward, epi_index)
    writer.add_scalar('train/entropy', entropy.item(), epi_index)
    writer.flush()

    # save model
    if max_acc_reward is None or acc_reward >= max_acc_reward:

        max_acc_reward = acc_reward

        savedir = os.path.join(outputdir, f'epi_{epi_index}_acc_reward={acc_reward:.2f}')
        os.makedirs(savedir)

        savepath = os.path.join(savedir, 'actor.pt')
        torch.save(actor.state_dict(), savepath)

        savepath = os.path.join(savedir, 'critic.pt')
        torch.save(critic.state_dict(), savepath)
