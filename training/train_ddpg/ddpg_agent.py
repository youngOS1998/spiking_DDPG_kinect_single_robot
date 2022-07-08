from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import os
import sys
sys.path.append('../../')
from training.train_ddpg.ddpg_networks import ActorNet, CriticNet


class Agent:
    """
    Class for DDPG Agent

    Main Function:
        1. Remember: Insert new memory into the memory list

        2. Act: Generate New Action base on actor network

        3. Replay: Train networks base on mini-batch replay

        4. Save: Save actor network weights

        5. Load: Load actor network weights
    """
    def __init__(self,
                 state_num,
                 action_num,
                 rescale_state_num,
                 actor_net_dim=(256, 256, 256),
                 critic_net_dim=(512, 512, 512),
                 memory_size=1000,
                 batch_size=128,
                 target_tau=0.01,
                 target_update_steps=5,
                 reward_gamma=0.99,
                 actor_lr=0.0001,
                 critic_lr=0.0001,
                 epsilon_start=0.9,
                 epsilon_end=0.01,
                 epsilon_decay=0.999,
                 epsilon_rand_decay_start=60000,
                 epsilon_rand_decay_step=1,
                 poisson_window=50,
                 use_poisson=False,
                 use_cuda=True,
                 in_channel = 1,
                 mid_channel = 3,
                 out_channel = 1):
        """

        :param state_num: number of state
        :param action_num: number of action
        :param rescale_state_num: number of rescale state
        :param actor_net_dim: dimension of actor network
        :param critic_net_dim: dimension of critic network
        :param memory_size: size of memory
        :param batch_size: size of mini-batch
        :param target_tau: update rate for target network
        :param target_update_steps: update steps for target network
        :param reward_gamma: decay of future reward
        :param actor_lr: learning rate for actor network
        :param critic_lr: learning rate for critic network
        :param epsilon_start: max value for random action
        :param epsilon_end: min value for random action
        :param epsilon_decay: steps from max to min random action
        :param epsilon_rand_decay_start: start step for epsilon start to decay
        :param epsilon_rand_decay_step: steps between epsilon decay
        :param poisson_window: window of poisson spike
        :param use_poisson: if or not use poisson spike random
        :param use_cuda: if or not use gpu
        """
        self.state_num = state_num
        self.action_num = action_num
        self.rescale_state_num = rescale_state_num
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.reward_gamma = reward_gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon_rand_decay_start = epsilon_rand_decay_start
        self.epsilon_rand_decay_step = epsilon_rand_decay_step
        self.poisson_window = poisson_window
        self.use_poisson = use_poisson
        self.use_cuda = use_cuda
        self.in_channel = in_channel
        self.mid_channel = mid_channel
        self.out_channel = out_channel
        '''
        Random Action
        '''
        self.epsilon = epsilon_start
        '''
        Device
        '''
        if self.use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        """
        Memory
        """
        self.memory = deque(maxlen=self.memory_size)
        """
        Networks and Target Networks
        """
        self.actor_net = ActorNet(state_num=4, action_num=2,
                                  hidden1=actor_net_dim[0],
                                  hidden2=actor_net_dim[1],
                                  hidden3=actor_net_dim[2],
                                  in_channel=self.in_channel,
                                  mid_channel=self.mid_channel,
                                  out_channel=self.out_channel)

        self.critic_net = CriticNet(state_num=4, action_num=2,
                                    hidden1=critic_net_dim[0],
                                    hidden2=critic_net_dim[1],
                                    hidden3=critic_net_dim[2],
                                    in_channel=self.in_channel,
                                    mid_channel=self.mid_channel,
                                    out_channel=self.out_channel)

        self.target_actor_net = ActorNet(state_num=4, action_num=2,
                                         hidden1=actor_net_dim[0],
                                         hidden2=actor_net_dim[1],
                                         hidden3=actor_net_dim[2],
                                         in_channel=self.in_channel,
                                         mid_channel=self.mid_channel,
                                         out_channel=self.out_channel)

        self.target_critic_net = CriticNet(state_num=4, action_num=2,
                                           hidden1=critic_net_dim[0],
                                           hidden2=critic_net_dim[1],
                                           hidden3=critic_net_dim[2],
                                           in_channel=self.in_channel,
                                           mid_channel=self.mid_channel,
                                           out_channel=self.out_channel)

        self._hard_update(self.target_actor_net, self.actor_net)
        self._hard_update(self.target_critic_net, self.critic_net)
        self.actor_net.to(self.device)
        self.critic_net.to(self.device)
        self.target_actor_net.to(self.device)
        self.target_critic_net.to(self.device)
        """
        Criterion and optimizers
        """
        self.criterion = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)
        """
        Step Counter
        """
        self.step_ita = 0

    def remember(self, state_depth, rescale_state_depth, action, reward, next_state_depth, rescale_next_state_depth, done):
        """
        Add New Memory Entry into memory deque
        :param state: current state
        :param action: current action
        :param reward: reward after action
        :param next_state: next action
        :param done: if is done
        """
        self.memory.append((state_depth, rescale_state_depth, action, reward, next_state_depth, rescale_next_state_depth, done))

    def act(self, state, explore=True, train=True):
        """
        Generate Action based on state
        ##:param state: current state
        :param state: [list(1x4), depth_state(np.array(1x480x640))]
        :param explore: if or not do random explore
        :param train: if or not in training
        :return: action
        """
        normal_state = state[0]   # list(1x4)
        depth_img = state[1]      # depth state(np.array(1x480x640))
        with torch.no_grad():
            normal_state_tmp = np.array(normal_state)
            depth_img_tmp = np.array(depth_img)
            if self.use_poisson:
                state = self._state_2_poisson_state(state, 1)
            normal_state = torch.Tensor(normal_state_tmp.reshape((1, -1))).to(self.device)      # 1 x 22
            depth_img = torch.Tensor(depth_img_tmp).unsqueeze(0).to(self.device)   # (1,1,480,640)
            #print("shape (depth): ", depth_img.shape)
            #print("shape (normal_state): ", normal_state.shape)
            action = self.actor_net([depth_img, normal_state]).to('cpu')
            action = action.numpy().squeeze()
        if train:
            if self.step_ita > self.epsilon_rand_decay_start and self.epsilon > self.epsilon_end:
                if self.step_ita % self.epsilon_rand_decay_step == 0:
                    self.epsilon = self.epsilon * self.epsilon_decay
            noise = np.random.randn(self.action_num) * self.epsilon
            action = noise + (1 - self.epsilon) * action
            action = np.clip(action, [0., 0.], [1., 1.])
        elif explore:
            noise = np.random.randn(self.action_num) * self.epsilon_end
            action = noise + (1 - self.epsilon_end) * action
            action = np.clip(action, [0., 0.], [1., 1.])
        return action.tolist()

    def replay(self):
        """
        Experience Replay Training
        :return: actor_loss_item, critic_loss_item
        """
        state_batch, depth_img_batch, r_state_batch, r_depth_batch, action_batch, reward_batch, \
            nstate_batch, ndepth_batch, r_nstate_batch, r_ndepth_batch, done_batch = self._random_minibatch()
        # (state, rescale_state, action, reward, next_state, rescale_next_state, done)
        '''
        Compuate Target Q Value
        '''
        with torch.no_grad():
            #print("shape(r_state_batch): ", r_state_batch.shape)
            #print("shape(r_depth_batch): ", r_depth_batch.shape)
            naction_batch = self.target_actor_net([r_ndepth_batch, r_nstate_batch])                     # 根据现在时刻状态rescale_
            ##next_q = self.target_critic_net([nstate_batch, naction_batch])
            next_q = self.target_critic_net([nstate_batch, ndepth_batch, naction_batch])
            target_q = reward_batch + self.reward_gamma * next_q * (1. - done_batch)  # TD-target
        '''
        Update Critic Network
        '''
        self.critic_optimizer.zero_grad()
        ##current_q = self.critic_net([state_batch, action_batch])
        current_q = self.critic_net([state_batch, depth_img_batch, action_batch])
        critic_loss = self.criterion(current_q, target_q)
        critic_loss_item = critic_loss.item()
        critic_loss.backward()
        self.critic_optimizer.step()
        '''
        Update Actor Network
        '''
        self.actor_optimizer.zero_grad()
        current_action = self.actor_net([r_depth_batch, r_state_batch])
        ##actor_loss = -self.critic_net([state_batch, current_action])
        actor_loss = -self.critic_net([state_batch, depth_img_batch, current_action])
        actor_loss = actor_loss.mean()
        actor_loss_item = actor_loss.item()
        actor_loss.backward()
        self.actor_optimizer.step()
        '''
        Update Target Networks
        '''
        self.step_ita += 1
        if self.step_ita % self.target_update_steps == 0:
            self._soft_update(self.target_actor_net, self.actor_net)
            self._soft_update(self.target_critic_net, self.critic_net)
        return actor_loss_item, critic_loss_item

    def reset_epsilon(self, new_epsilon, new_decay):
        """
        Set Epsilon to a new value
        :param new_epsilon: new epsilon value
        :param new_decay: new epsilon decay
        """
        self.epsilon = new_epsilon
        self.epsilon_decay = new_decay

    def save(self, save_dir, episode, run_name):
        """
        Save Actor Net weights
        :param save_dir: directory for saving weights
        :param episode: number of episode
        :param run_name: name of the run
        """
        try:
            os.mkdir(save_dir)
            print("Directory ", save_dir, " Created")
        except FileExistsError:
            print("Directory", save_dir, " already exists")
        torch.save(self.actor_net.state_dict(),
                   save_dir + '/' + run_name + '_actor_network_s' + str(episode) + '.pt')
        torch.save(self.actor_net.net1.state_dict(),
                   save_dir + '/' + run_name + '_cnn_network_s' + str(episode) + '.pt')
        print("Episode " + str(episode) + " weights saved ...")

    def load(self, load_file_name):
        """
        Load Actor Net weights
        :param load_file_name: weights file name
        """
        self.actor_net.to('cpu')
        self.actor_net.load_state_dict(torch.load(load_file_name))
        self.actor_net.to(self.device)

    def _state_2_poisson_state(self, state_value, batch_size):
        """
        Transform state to spikes then transform back to state to add random
        :param state_value: state from environment transfer to firing rates of neurons
        :param batch_size: batch size
        :return: poisson_state
        """
        spike_state_value = state_value.reshape((batch_size, self.rescale_state_num, 1))
        state_spikes = np.random.rand(batch_size, self.rescale_state_num, self.poisson_window) < spike_state_value
        poisson_state = np.sum(state_spikes, axis=2).reshape((batch_size, -1))
        poisson_state = poisson_state / self.poisson_window
        poisson_state = poisson_state.astype(float)
        return poisson_state

    def _random_minibatch(self):
        """
        Random select mini-batch from memory
        :return: state_batch, action_batch, reward_batch, nstate_batch, done_batch
        """
        minibatch = random.sample(self.memory, self.batch_size)
        depth_img_batch = np.zeros((self.batch_size, 1, 48, 64))
        state_batch = np.zeros((self.batch_size, self.state_num))
        rescale_state_batch = np.zeros((self.batch_size, 4))
        rescale_depth_batch = np.zeros((self.batch_size, 1, 48, 64))
        action_batch = np.zeros((self.batch_size, self.action_num))
        reward_batch = np.zeros((self.batch_size, 1))
        nstate_batch = np.zeros((self.batch_size, self.state_num))
        ndepth_batch = np.zeros((self.batch_size, 1, 48, 64))
        rescale_nstate_batch = np.zeros((self.batch_size, 4))
        rescale_ndepth_batch = np.zeros((self.batch_size, 1, 48, 64))
        done_batch = np.zeros((self.batch_size, 1))
        for num in range(self.batch_size):
            state_batch[num, :] = np.array(minibatch[num][0][0])        # list(1x4)
            depth_img_batch[num, :] = np.array(minibatch[num][0][1])    # np.array(1x480x640)
            rescale_state_batch[num, :] = np.array(minibatch[num][1][0])
            rescale_depth_batch[num, :] = np.array(minibatch[num][1][1])
            action_batch[num, :] = np.array(minibatch[num][2])
            reward_batch[num, 0] = minibatch[num][3]
            nstate_batch[num, :] = np.array(minibatch[num][4][0])
            ndepth_batch[num, :] = np.array(minibatch[num][4][1])
            rescale_nstate_batch[num, :] = np.array(minibatch[num][5][0])
            rescale_ndepth_batch[num, :] = np.array(minibatch[num][5][1])
            done_batch[num, 0] = minibatch[num][6]
        if self.use_poisson:
            rescale_state_batch = self._state_2_poisson_state(rescale_state_batch, self.batch_size)
            rescale_nstate_batch = self._state_2_poisson_state(rescale_nstate_batch, self.batch_size)
        state_batch = torch.Tensor(state_batch).to(self.device)
        depth_img_batch = torch.Tensor(depth_img_batch).to(self.device)
        rescale_state_batch = torch.Tensor(rescale_state_batch).to(self.device)
        rescale_depth_batch = torch.Tensor(rescale_depth_batch).to(self.device)
        action_batch = torch.Tensor(action_batch).to(self.device)
        reward_batch = torch.Tensor(reward_batch).to(self.device)
        nstate_batch = torch.Tensor(nstate_batch).to(self.device)
        ndepth_batch = torch.Tensor(ndepth_batch).to(self.device)
        rescale_nstate_batch = torch.Tensor(rescale_nstate_batch).to(self.device)
        rescale_ndepth_batch = torch.Tensor(rescale_ndepth_batch).to(self.device)
        done_batch = torch.Tensor(done_batch).to(self.device)
        return state_batch, depth_img_batch, rescale_state_batch, rescale_depth_batch, action_batch, \
               reward_batch, nstate_batch, ndepth_batch, rescale_nstate_batch, rescale_ndepth_batch, done_batch   # 是一个batch_size的数据

    def _hard_update(self, target, source):
        """
        Hard Update Weights from source network to target network
        :param target: target network
        :param source: source network
        """
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

    def _soft_update(self, target, source):
        """
        Soft Update weights from source network to target network
        :param target: target network
        :param source: source network
        """
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.target_tau) + param.data * self.target_tau
                )
