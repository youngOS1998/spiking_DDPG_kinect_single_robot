from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import numpy as np

"""
Our implementation of STBP on SNN is inspired by 
the open-sourced implementation of STBP for:

Wu, Yujie, Lei Deng, Guoqi Li, Jun Zhu, and Luping Shi. 
"Spatio-temporal backpropagation for training high-performance spiking neural networks." 
Frontiers in neuroscience 12 (2018).

Their implementation for SCNN can be found here:
https://github.com/yjwu17/BP-for-SpikingNN

We would like to thank them for open-source their implementation.
"""


NEURON_VTH = 0.5
NEURON_CDECAY = 0.5
NEURON_VDECAY = 0.75
SPIKE_PSEUDO_GRAD_WINDOW = 0.5


class PseudoSpikeRect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(NEURON_VTH).float()           # 大于 NEURON_VTH 就发放脉冲
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        spike_pseudo_grad = (abs(input - NEURON_VTH) < SPIKE_PSEUDO_GRAD_WINDOW)
        return grad_input * spike_pseudo_grad.float()


class ActorNetSpiking(nn.Module):
    """ Spiking Actor Network """
    def __init__(self, combined_num, action_num, device, batch_window=5, hidden1=4800, hidden2=2400, hidden3=1200, hidden4=256):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param device: device used
        :param batch_window: window steps
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(ActorNetSpiking, self).__init__()
        self.combined_num = combined_num   # 9602
        self.action_num = action_num
        self.device = device
        self.batch_window = batch_window
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.pseudo_spike = PseudoSpikeRect.apply
        # conv2 for positive events
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(4, 4), stride=2, padding=1)    # 3 x 240 x 320
        self.conv1_2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(4, 4), stride=2, padding=1)   # 10 x 120 x 160
        self.conv1_3 = nn.Conv2d(in_channels=10, out_channels=4, kernel_size=(4, 4), stride=4, padding=0)   # 4 x 30 x 40
        # conv2 for negative events
        self.conv2_1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(4, 4), stride=2, padding=1)    # 3 x 240 x 320
        self.conv2_2 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(4, 4), stride=2, padding=1)   # 10 x 120 x 160
        self.conv2_3 = nn.Conv2d(in_channels=10, out_channels=4, kernel_size=(4, 4), stride=4, padding=0)   # 4 x 30 x 40

        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()

        self.fc1 = nn.Linear(self.combined_num, self.hidden1, bias=True)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=True)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3, bias=True)
        self.fc4 = nn.Linear(self.hidden3, self.hidden4, bias=True)
        self.fc5 = nn.Linear(self.hidden4, self.action_num, bias=True)


    def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
        """
        Neuron Model
        :param syn_func: synaptic function
        :param pre_layer_output: output from pre-synaptic layer
        :param current: current of last step
        :param volt: voltage of last step
        :param spike: spike of last step
        :return: current, volt, spike
        """
        current = current * NEURON_CDECAY + syn_func(pre_layer_output)
        volt = volt * NEURON_VDECAY * (1. - spike) + current
        spike = self.pseudo_spike(volt)
        return current, volt, spike

    def normal_state_2_spikes(self, normal_state): 
        normal_state = normal_state[np.newaxis, :]   # time_step x batch_size x normal_num
        batch_size, normal_num = normal_state.shape[1], normal_state.shape[2]
        normal_spikes = np.random.rand(self.batch_window, batch_size, normal_num) < normal_state
        normal_spikes = normal_spikes.astype(float)
        normal_spikes = torch.Tensor(normal_spikes).to(self.device)
        return normal_spikes


    def forward(self, x, batch_size):   # x/state: [list(1x6), np.array(1x48x64)]
                                        # x: 
        """
        :param x: state batch
        :param batch_size: size of batch
        :return: out
        """
        cv1_1_u = torch.zeros(batch_size, 3, 240, 320, device=self.device)
        cv1_1_v = torch.zeros(batch_size, 3, 240, 320, device=self.device)
        cv1_1_s = torch.zeros(batch_size, 3, 240, 320, device=self.device)

        cv1_2_u = torch.zeros(batch_size, 10, 120, 160, device=self.device)
        cv1_2_v = torch.zeros(batch_size, 10, 120, 160, device=self.device)
        cv1_2_s = torch.zeros(batch_size, 10, 120, 160, device=self.device)

        cv1_3_u = torch.zeros(batch_size, 4, 30, 40, device=self.device)
        cv1_3_v = torch.zeros(batch_size, 4, 30, 40, device=self.device)
        cv1_3_s = torch.zeros(batch_size, 4, 30, 40, device=self.device)

        cv2_1_u = torch.zeros(batch_size, 3, 240, 320, device=self.device)
        cv2_1_v = torch.zeros(batch_size, 3, 240, 320, device=self.device)
        cv2_1_s = torch.zeros(batch_size, 3, 240, 320, device=self.device)

        cv2_2_u = torch.zeros(batch_size, 10, 120, 160, device=self.device)
        cv2_2_v = torch.zeros(batch_size, 10, 120, 160, device=self.device)
        cv2_2_s = torch.zeros(batch_size, 10, 120, 160, device=self.device)

        cv2_3_u = torch.zeros(batch_size, 4, 30, 40, device=self.device)
        cv2_3_v = torch.zeros(batch_size, 4, 30, 40, device=self.device)
        cv2_3_s = torch.zeros(batch_size, 4, 30, 40, device=self.device)      

        fc1_u = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_v = torch.zeros(batch_size, self.hidden1, device=self.device)
        fc1_s = torch.zeros(batch_size, self.hidden1, device=self.device)

        fc2_u = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc2_v = torch.zeros(batch_size, self.hidden2, device=self.device)
        fc2_s = torch.zeros(batch_size, self.hidden2, device=self.device)
        
        fc3_u = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc3_v = torch.zeros(batch_size, self.hidden3, device=self.device)
        fc3_s = torch.zeros(batch_size, self.hidden3, device=self.device)
        
        fc4_u = torch.zeros(batch_size, self.hidden4, device=self.device)
        fc4_v = torch.zeros(batch_size, self.hidden4, device=self.device)
        fc4_s = torch.zeros(batch_size, self.hidden4, device=self.device)
        
        fc5_u = torch.zeros(batch_size, self.action_num, device=self.device)
        fc5_v = torch.zeros(batch_size, self.action_num, device=self.device)
        fc5_s = torch.zeros(batch_size, self.action_num, device=self.device)       
        fc5_sumspike = torch.zeros(batch_size, self.action_num, device=self.device)

        # propose the normal state to spiking state
        normal_state = x[2]
        normal_spikes = self.normal_state_2_spikes(normal_state)

        for step in range(self.batch_window):

          # input_spike_pos, input_spike_neg = x[:, :, step]    # input_spike: tensor: batch_size x 196
            input_spike_pos, input_spike_neg = x[0][step, :, :, :, :], x[1][step, :, :, :, :]    # step x batch_size x channels x height x width
            normal_spikes_input = normal_spikes[step, :, :]
            cv1_1_u, cv1_1_v, cv1_1_s = self.neuron_model(self.conv1_1, input_spike_pos, cv1_1_u, cv1_1_v, cv1_1_s)
            cv1_2_u, cv1_2_v, cv1_2_s = self.neuron_model(self.conv1_2, cv1_1_s, cv1_2_u, cv1_2_v, cv1_2_s)
            cv1_3_u, cv1_3_v, cv1_3_s = self.neuron_model(self.conv1_3, cv1_2_s, cv1_3_u, cv1_3_v, cv1_3_s)
            output_pos = self.flatten1(cv1_3_s)
            print(output_pos.shape)

            cv2_1_u, cv2_1_v, cv2_1_s = self.neuron_model(self.conv2_1, input_spike_neg, cv2_1_u, cv2_1_v, cv2_1_s)
            cv2_2_u, cv2_2_v, cv2_2_s = self.neuron_model(self.conv2_2, cv2_1_s, cv2_2_u, cv2_2_v, cv2_2_s)
            cv2_3_u, cv2_3_v, cv2_3_s = self.neuron_model(self.conv2_3, cv2_2_s, cv2_3_u, cv2_3_v, cv2_3_s)           
            output_neg = self.flatten2(cv2_3_s)
            print(output_neg.shape)
            print(normal_spikes.shape)

            combined_data = torch.cat([output_pos, output_neg, normal_spikes_input], axis=1)   # batch x 9602   
            fc1_u, fc1_v, fc1_s = self.neuron_model(self.fc1, combined_data, fc1_u, fc1_v, fc1_s)
            fc2_u, fc2_v, fc2_s = self.neuron_model(self.fc2, fc1_s, fc2_u, fc2_v, fc2_s)
            fc3_u, fc3_v, fc3_s = self.neuron_model(self.fc3, fc2_s, fc3_u, fc3_v, fc3_s)
            fc4_u, fc4_v, fc4_s = self.neuron_model(self.fc4, fc3_s, fc4_u, fc4_v, fc4_s)
            fc5_u, fc5_v, fc5_s = self.neuron_model(self.fc5, fc4_s, fc5_u, fc5_v, fc5_s)
            fc5_sumspike += fc5_s
        out = fc5_sumspike / self.batch_window
        return out  #  tensor: batch_size x 2



class CriticNetSpiking(nn.Module): 
    """ Critic Network"""
    def __init__(self, normal_state_num, action_num, hidden1=512, hidden2=512, hidden3=512, in_channel=1, mid_channel=3, out_channel=1):
        """
        :param state_num: number of states
        :param action_num: number of actions
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(CriticNetSpiking, self).__init__()

        self.net_pos = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=10, kernel_size=(4, 4), stride=2, padding=1),        # 10 x 240 x 320
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=4, kernel_size=(4, 4), stride=2, padding=1),        # 4 x 120 x 160
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=(4, 4), stride=4, padding=0),         # 3 x 30 x 40
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=2, padding=0),         # 1 x 14 x 19
            nn.ReLU(),
            nn.Flatten()
        )

        self.net_neg = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=10, kernel_size=(4, 4), stride=2, padding=1),        # 10 x 240 x 320
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=4, kernel_size=(4, 4), stride=2, padding=1),        # 4 x 120 x 160
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=3, kernel_size=(4, 4), stride=4, padding=0),         # 3 x 30 x 40
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3, 3), stride=2, padding=0),         # 1 x 14 x 19
            nn.ReLU(),
            nn.Flatten()
        )

        self.net2 = nn.Sequential(
            nn.Linear(normal_state_num + 266 + 266 + action_num, hidden1),  # 4 + 266 + 266 + 2
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, 1)
        )

    def forward(self, x):
        events_pos, events_neg, normal_state, action_state = x[0], x[1], x[2], x[3]
        feature_pos, feature_neg = self.net_pos(events_pos), self.net_neg(events_neg)  # 1 x 266
        data_in = torch.cat([feature_pos, feature_neg, normal_state, action_state], 1) # 1 x (266 + 266 + 2 + 2) 
        out = self.net2(data_in)
        return out


if __name__ == "__main__":

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda')
    actor_net = ActorNetSpiking(combined_num=9604, action_num=2, device=device).to(device)
    # critic_net = CriticNetSpiking(normal_state_num=2, action_num=2).to(device)

    # # evaluation for critic network
    # with torch.no_grad():
    #     spike_value_pos = np.random.rand(1, 1, 480, 640)
    #     spike_value_neg = np.random.rand(1, 1, 480, 640)
    #     data_pos = np.random.rand(5, 6, 480, 640) < spike_value_pos  # batch_size x channels x height x width
    #     data_neg = np.random.rand(5, 6, 480, 640) < spike_value_neg
    #     state_spike_pos = data_pos.astype(float)
    #     state_spike_neg = data_neg.astype(float)
    #     state_spike_pos = torch.Tensor(state_spike_pos).to(device)
    #     state_spike_neg = torch.Tensor(state_spike_neg).to(device)
    #     normal_state = np.random.rand(5, 2)  # batch_size x normal_num
    #     normal_state = torch.Tensor(normal_state).to(device)
    #     action_state = np.random.rand(5, 2)  # batch_size x action_num
    #     action_state = torch.Tensor(action_state).to(device)
    #     xa = [state_spike_pos, state_spike_neg, normal_state, action_state]
    #     output = critic_net(xa)
    #     print(output)

    # evaluation for actor network
    with torch.no_grad():
        spike_value_pos = np.random.rand(1, 4, 1, 480, 640)
        spike_value_neg = np.random.rand(1, 4, 1, 480, 640)
        data_pos = np.random.rand(5, 4, 1, 480, 640) < spike_value_pos
        data_neg = np.random.rand(5, 4, 1, 480, 640) < spike_value_neg
        state_spikes_pos = data_pos.astype(float)
        state_spikes_neg = data_neg.astype(float)
        state_spikes_pos = torch.Tensor(state_spikes_pos).to(device)
        state_spikes_neg = torch.Tensor(state_spikes_neg).to(device)
        normal_state = np.random.rand(4, 4)  # batch_size x state_num
        combined_data = [state_spikes_pos, state_spikes_neg, normal_state]
        output_data = actor_net(combined_data, 4)
    print(output_data)
    print(output_data.shape)
