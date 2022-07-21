from collections import deque
import pickle
import copy
import os
import random
import math
from typing import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('../../')
from training.train_ddpg.ddpg_networks import CriticNet
from training.train_spiking_ddpg.sddpg_networks import ActorNetSpiking, CriticNetSpiking


class AgentSpiking:
    """
    Class for DDPG Agent for Spiking Actor Network

    Main Function:
        1. Remember: Insert new memory into the memory list

        2. Act: Generate New Action base on actor network

        3. Replay: Train networks base on mini-batch replay

        4. Save: Save model
    """
    def __init__(self,
                 state_num=4,
                 action_num=2,
                 spike_state_num=9604,
                 actor_net_dim=(256, 256, 256),
                 critic_net_dim=(512, 512, 512),
                 batch_window=50,
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
                 use_cuda=True):
        """

        :param state_num: number of state
        :param action_num: number of action
        :param spike_state_num: number of state for spike actor
        :param actor_net_dim: dimension of actor network
        :param critic_net_dim: dimension of critic network
        :param batch_window: window steps for one sample
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
        :param use_cuda: if or not use gpu
        """
        self.state_num = state_num
        self.action_num = action_num
        self.spike_state_num = spike_state_num
        self.batch_window = batch_window
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
        self.use_cuda = use_cuda
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
        self.actor_net = ActorNetSpiking(self.spike_state_num, 
                                         self.action_num, 
                                         self.device,
                                         hidden1=actor_net_dim[0],
                                         hidden2=actor_net_dim[1],
                                         hidden3=actor_net_dim[2],
                                         batch_window=self.batch_window)

        self.critic_net = CriticNetSpiking(self.state_num, 
                                           self.action_num,
                                           hidden1=critic_net_dim[0],
                                           hidden2=critic_net_dim[1],
                                           hidden3=critic_net_dim[2])

        self.target_actor_net = ActorNetSpiking(self.spike_state_num, 
                                                self.action_num, 
                                                self.device,
                                                hidden1=actor_net_dim[0],
                                                hidden2=actor_net_dim[1],
                                                hidden3=actor_net_dim[2],
                                                batch_window=self.batch_window)

        self.target_critic_net = CriticNetSpiking(self.state_num, 
                                                  self.action_num,
                                                  hidden1=critic_net_dim[0],
                                                  hidden2=critic_net_dim[1],
                                                  hidden3=critic_net_dim[2])

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


    def remember(self, state, raw_action, reward, next_state, done):
        """
        Add New Memory Entry into memory deque
        :param state: current state
        :param spike_state: current state with separate neg and pos values
        :param action: current action
        :param reward: reward after action
        :param next_state: next state
        :param next_spike_state: next with separate neg and pos values
        :param done: if is done
        """
        
        self.memory.append((state, raw_action, reward, next_state, done))

    def act(self, state, explore=True, train=True):  # state: [list(1x4), [np.array(6x480x640), np.array(6x480x640)]
        """
        Generate Action based on state
        :param state: current state
        :param explore: if or not do random explore
        :param train: if or not in training
        :return: action
        """

        normal_state, dvs_state = state[0], state[1]
        with torch.no_grad():
            # from np: batch_size x state_num --> batch_window x batch_size x state_num
            # dvs_state: [np(6x480x640), np(6x480x640)]
            dvs_state = self._resort_dvs_state(dvs_state)  # return:  [np(step_window x batch_size x channels x height x width), np(step_window x batch_size x channels x height x width)]
            dvs_pos_state = dvs_state[0]
            dvs_neg_state = dvs_state[1]
            dvs_pos_state_tensor = torch.Tensor(dvs_pos_state).to(self.device)
            dvs_neg_state_tensor = torch.Tensor(dvs_neg_state).to(self.device)
            combined_data_actor = [dvs_pos_state_tensor, dvs_neg_state_tensor, normal_state]
            action = self.actor_net(combined_data_actor, 1).to(self.device)   # action: tensor: batch_size x 2
            action = action.cpu().numpy().squeeze()  # np: [1 x 2]
            print('action shape:',action.shape)
            raw_snn_action = copy.deepcopy(action)
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
        return action.tolist(), raw_snn_action.tolist()

    def _from_minibatch_to_replay(self, state):  # from np: batch_size x 6 x 480 x 640 --> np: batch_window x batch_size x channels x height x width
        """
        We need to transform the data from minibatch to replay
        """
        tmp_state = state.transpose(1,0,2,3)   # 6 x batch_size x 480 x 640
        tmp_state = tmp_state[:,:,np.newaxis,:,:]
        
        return tmp_state


    def replay(self):
        """
        Experience Replay Training
        :return: actor_loss_item, critic_loss_item
        """
        normal_state_batch, dvs_pos_state_batch, dvs_neg_state_batch, normal_nstate_batch, \
            dvs_pos_nstate_batch, dvs_neg_nstate_batch, action_batch, reward_batch, done_batch = self._random_minibatch()
        '''
        :param: normal_state_batch  : np: batch_size x 4
        :param: dvs_pos_state_batch : np: batch_size x 6 x 480 x 640
        :param: dvs_neg_state_batch : np: batch_size x 6 x 480 x 640
        :param: normal_nstate_batch : np: batch_szie x 4
        :param: dvs_pos_nstate_batch: np: batch_size x 6 x 480 x 640
        :param: dvs_neg_nstate_batch: np: batch_size x 6 x 480 x 640
        :param: action_batch        : np: batch_size x 2
        :param: reward_batch        : np: batch_size x 1
        :param: done_batch          : np: batch_size x 1

        NEED:
        :param: dvs_pos_state_batch : np: batch_window x batch_size x channels x height x width
        Compuate Target Q Value
        '''

        dvs_pos_state_batch_actor = self._from_minibatch_to_replay(dvs_pos_state_batch)
        dvs_neg_state_batch_actor = self._from_minibatch_to_replay(dvs_neg_state_batch)
        dvs_pos_nstate_batch_actor = self._from_minibatch_to_replay(dvs_pos_nstate_batch)
        dvs_neg_nstate_batch_actor = self._from_minibatch_to_replay(dvs_neg_nstate_batch)

        dvs_pos_state_batch_tensor = torch.Tensor(dvs_pos_state_batch).to(self.device)
        dvs_neg_state_batch_tensor = torch.Tensor(dvs_neg_state_batch).to(self.device)
        dvs_pos_nstate_batch_tensor = torch.Tensor(dvs_pos_nstate_batch).to(self.device)
        dvs_neg_nstate_batch_tensor = torch.Tensor(dvs_neg_nstate_batch).to(self.device)

        dvs_pos_state_batch_actor_tensor = torch.Tensor(dvs_pos_state_batch_actor).to(self.device)
        dvs_neg_state_batch_actor_tensor = torch.Tensor(dvs_neg_state_batch_actor).to(self.device)
        dvs_pos_nstate_batch_actor_tensor = torch.Tensor(dvs_pos_nstate_batch_actor).to(self.device)
        dvs_neg_nstate_batch_actor_tensor = torch.Tensor(dvs_neg_nstate_batch_actor).to(self.device)

        normal_state_batch_tensor = torch.Tensor(normal_state_batch).to(self.device)
        normal_nstate_batch_tensor = torch.Tensor(normal_nstate_batch).to(self.device)
        action_batch_tensor = torch.Tensor(action_batch).to(self.device)
        
        combined_data_actor = [dvs_pos_state_batch_actor_tensor, dvs_neg_state_batch_actor_tensor, normal_state_batch]
        
        combined_ndata_actor = [dvs_pos_nstate_batch_actor_tensor, dvs_neg_nstate_batch_actor_tensor, normal_nstate_batch]
        

        with torch.no_grad():
            reward_batch = torch.Tensor(reward_batch).to(self.device)
            done_batch = torch.Tensor(done_batch).to(self.device)
            naction_batch = self.target_actor_net(combined_ndata_actor, self.batch_size)
            combined_ndata_critic = [dvs_pos_nstate_batch_tensor, dvs_neg_nstate_batch_tensor, \
                                     normal_nstate_batch_tensor, naction_batch]
            next_q = self.target_critic_net(combined_ndata_critic)
            target_q = reward_batch + self.reward_gamma * next_q * (1. - done_batch)
        '''
        Update Critic Network
        '''
        combined_data_critic = [dvs_pos_state_batch_tensor, dvs_neg_state_batch_tensor, \
                                normal_state_batch_tensor, action_batch_tensor]
        self.critic_optimizer.zero_grad()
        current_q = self.critic_net(combined_data_critic)
        critic_loss = self.criterion(current_q, target_q)
        critic_loss_item = critic_loss.item()
        critic_loss.backward()
        self.critic_optimizer.step()
        '''
        Update Actor Network
        '''
        self.actor_optimizer.zero_grad()
        current_action = self.actor_net(combined_data_actor, self.batch_size)
        combined_data_critic = [dvs_pos_state_batch_tensor, dvs_neg_state_batch_tensor, \
                                normal_state_batch_tensor, current_action]
        actor_loss = -self.critic_net(combined_data_critic)
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
        Save SNN Actor Net weights
        :param save_dir: directory for saving weights
        :param episode: number of episode
        :return: max_w, min_w, max_bias, min_bias, shape_w, shape_bias
        """
        try:
            os.mkdir(save_dir)
            print("Directory ", save_dir, " Created")
        except FileExistsError:
            print("Directory", save_dir, " already exists")
        self.actor_net.to('cpu')
        l1_weights = self.actor_net.fc1.weight.data.numpy()
        l1_bias = self.actor_net.fc1.bias.data.numpy()
        l2_weights = self.actor_net.fc2.weight.data.numpy()
        l2_bias = self.actor_net.fc2.bias.data.numpy()
        l3_weights = self.actor_net.fc3.weight.data.numpy()
        l3_bias = self.actor_net.fc3.bias.data.numpy()
        l4_weights = self.actor_net.fc4.weight.data.numpy()
        l4_bias = self.actor_net.fc4.bias.data.numpy()
        pickle.dump([l1_weights, l2_weights, l3_weights, l4_weights],
                    open(save_dir + '/' + run_name + '_snn_weights_s' + str(episode) + '.p', 'wb+'))
        pickle.dump([l1_bias, l2_bias, l3_bias, l4_bias],
                    open(save_dir + '/' + run_name + '_snn_bias_s' + str(episode) + '.p', 'wb+'))
        l1_max = np.amax(l1_weights)
        l1_min = np.amin(l1_weights)
        l1_shape = l1_weights.shape
        l1_bias_max = np.amax(l1_bias)
        l1_bias_min = np.amin(l1_bias)
        l1_bias_shape = l1_bias.shape
        l2_max = np.amax(l2_weights)
        l2_min = np.amin(l2_weights)
        l2_shape = l2_weights.shape
        l2_bias_max = np.amax(l2_bias)
        l2_bias_min = np.amin(l2_bias)
        l2_bias_shape = l2_bias.shape
        l3_max = np.amax(l3_weights)
        l3_min = np.amin(l3_weights)
        l3_shape = l3_weights.shape
        l3_bias_max = np.amax(l3_bias)
        l3_bias_min = np.amin(l3_bias)
        l3_bias_shape = l3_bias.shape
        l4_max = np.amax(l4_weights)
        l4_min = np.amin(l4_weights)
        l4_shape = l4_weights.shape
        l4_bias_max = np.amax(l4_bias)
        l4_bias_min = np.amin(l4_bias)
        l4_bias_shape = l4_bias.shape
        max_w = (l1_max, l2_max, l3_max, l4_max)
        min_w = (l1_min, l2_min, l3_min, l4_min)
        shape_w = (l1_shape, l2_shape, l3_shape, l4_shape)
        max_bias = (l1_bias_max, l2_bias_max, l3_bias_max, l4_bias_max)
        min_bias = (l1_bias_min, l2_bias_min, l3_bias_min, l4_bias_min)
        shape_bias = (l1_bias_shape, l2_bias_shape, l3_bias_shape, l4_bias_shape)
        self.actor_net.to(self.device)
        return max_w, min_w, max_bias, min_bias, shape_w, shape_bias

    def _resort_dvs_state(self, dvs_state):
        """
        Transform dvs_state to resorted spikes
        :param dvs_state: [np(6x480x640), np(6x480x640)]
        :return:  [np(step_window x batch_size x channels x height x width), np(step_window x batch_size x channels x height x width)]
        """
        dvs_pos = dvs_state[0]
        dvs_neg = dvs_state[1]  # np: 6 x 480 x 640 : step_window x height x width
        dvs_pos_sorted = dvs_pos[:,np.newaxis,np.newaxis,:,:]
        dvs_neg_sorted = dvs_neg[:,np.newaxis,np.newaxis,:,:]
        return [dvs_pos_sorted, dvs_neg_sorted]

    def _state_2_state_spikes(self, spike_state_value, batch_size):  # from np: batch_size x state_num --> batch_window x batch_size x state_num
        """
        Transform state to spikes of input neurons
        :param spike_state_value: state from environment transfer to firing rates of neurons
        :param batch_size: batch size
        :return: state_spikes
        """
        # spike_state_value = spike_state_value.reshape((-1, self.spike_state_num, 1)) # np: [1x198x1]
        spike_state_value = spike_state_value.reshape((1, -1, self.spike_state_num)) # np: [1 x batch_size x 198]
        # state_spikes = np.random.rand(batch_size, self.spike_state_num, self.batch_window) < spike_state_value
        state_spikes = np.random.rand(self.batch_window, batch_size, self.spike_state_num) < spike_state_value
        state_spikes = state_spikes.astype(float)
        state_spikes = torch.Tensor(state_spikes).to(self.device)
        return state_spikes  # batch_window x batch_size x spike_state_num

    def _random_minibatch(self):
        """
        Random select mini-batch from memory
        :return action_batch, reward_batch, done_batch, state_spikes_batch, nstate_spikes_batch
        """
        minibatch = random.sample(self.memory, self.batch_size)
        normal_state_batch = np.zeros((self.batch_size, 4))
        dvs_pos_state_batch = np.zeros((self.batch_size, 6, 480, 640))
        dvs_neg_state_batch = np.zeros((self.batch_size, 6, 480, 640))
        action_batch = np.zeros((self.batch_size, self.action_num))
        reward_batch = np.zeros((self.batch_size, 1))
        normal_nstate_batch = np.zeros((self.batch_size, 4))
        dvs_pos_nstate_batch = np.zeros((self.batch_size, 6, 480, 640))
        dvs_neg_nstate_batch = np.zeros((self.batch_size, 6, 480, 640))
        done_batch = np.zeros((self.batch_size, 1))
        for num in range(self.batch_size):
            normal_state_batch[num, :] = minibatch[num][0][0]
            dvs_pos_state_batch[num, :] = minibatch[num][0][1][0]
            dvs_neg_state_batch[num, :] = minibatch[num][0][1][1]
            tmp_data = np.array(minibatch[num][1])
            #print('Here:', tmp_data.shape)
            action_batch[num, :] = np.array(minibatch[num][1])
            reward_batch[num, 0] = minibatch[num][2]
            normal_nstate_batch[num, :] = minibatch[num][3][0]
            dvs_pos_nstate_batch[num, :] = minibatch[num][3][1][0]
            dvs_neg_nstate_batch[num, :] = minibatch[num][3][1][1]
            done_batch[num, 0] = minibatch[num][4]
        # state_spikes_batch = spike_state_value_batch
        # nstate_spikes_batch = spike_nstate_value_batch
        # action_batch = torch.Tensor(action_batch).to(self.device)
        # reward_batch = torch.Tensor(reward_batch).to(self.device)
        # done_batch = torch.Tensor(done_batch).to(self.device)
        return normal_state_batch, dvs_pos_state_batch, dvs_neg_state_batch, normal_nstate_batch, \
               dvs_pos_nstate_batch, dvs_neg_nstate_batch, action_batch, reward_batch, done_batch

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
