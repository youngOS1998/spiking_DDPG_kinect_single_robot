# import torch
# import torch.nn as nn
# import numpy as np

# class PseudoSpikeRect(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return input.gt(0.5).float()           # 大于 NEURON_VTH 就发放脉冲
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         spike_pseudo_grad = (abs(input - 0.5) < 0.5)
#         return grad_input * spike_pseudo_grad.float()

# class ActorNetSpiking(nn.Module):
#     """ Spiking Actor Network """
#     def __init__(self, state_num, action_num, device, batch_window=10, hidden1=256, hidden2=256, hidden3=256, \
#                 in_channel=1, mid_channel=3, out_channel=1
#         ):
#         """

#         :param state_num: number of states
#         :param action_num: number of actions
#         :param device: device used
#         :param batch_window: window steps
#         :param hidden1: hidden layer 1 dimension
#         :param hidden2: hidden layer 2 dimension
#         :param hidden3: hidden layer 3 dimension
#         """
#         super(ActorNetSpiking, self).__init__()
#         self.state_num = state_num
#         self.action_num = action_num
#         self.device = device
#         self.batch_window = batch_window
#         self.hidden1 = hidden1
#         self.hidden2 = hidden2
#         self.hidden3 = hidden3
#         self.pseudo_spike = PseudoSpikeRect.apply
#         self.fc1 = nn.Linear(self.state_num, self.hidden1, bias=True)
#         self.fc2 = nn.Linear(self.hidden1, self.hidden2, bias=True)
#         self.fc3 = nn.Linear(self.hidden2, self.hidden3, bias=True)
#         self.fc4 = nn.Linear(self.hidden3, self.action_num, bias=True)

#     def neuron_model(self, syn_func, pre_layer_output, current, volt, spike):
#         """
#         Neuron Model
#         :param syn_func: synaptic function
#         :param pre_layer_output: output from pre-synaptic layer
#         :param current: current of last step
#         :param volt: voltage of last step
#         :param spike: spike of last step
#         :return: current, volt, spike
#         """
#         current = current * 0.5 + syn_func(pre_layer_output)
#         volt = volt * 0.75 * (1. - spike) + current
#         spike = self.pseudo_spike(volt)
#         return current, volt, spike

#     def forward(self, x, batch_size):
#         """

#         :param x: state batch
#         :param batch_size: size of batch
#         :return: out
#         """

#         fc1_u = torch.zeros(batch_size, self.hidden1, device=self.device)
#         fc1_v = torch.zeros(batch_size, self.hidden1, device=self.device)
#         fc1_s = torch.zeros(batch_size, self.hidden1, device=self.device)
#         fc2_u = torch.zeros(batch_size, self.hidden2, device=self.device)
#         fc2_v = torch.zeros(batch_size, self.hidden2, device=self.device)
#         fc2_s = torch.zeros(batch_size, self.hidden2, device=self.device)
#         fc3_u = torch.zeros(batch_size, self.hidden3, device=self.device)
#         fc3_v = torch.zeros(batch_size, self.hidden3, device=self.device)
#         fc3_s = torch.zeros(batch_size, self.hidden3, device=self.device)
#         fc4_u = torch.zeros(batch_size, self.action_num, device=self.device)
#         fc4_v = torch.zeros(batch_size, self.action_num, device=self.device)
#         fc4_s = torch.zeros(batch_size, self.action_num, device=self.device)
#         fc4_sumspike = torch.zeros(batch_size, self.action_num, device=self.device)
#         for step in range(self.batch_window):
#             input_spike = x[:, :, step]
#             fc1_u, fc1_v, fc1_s = self.neuron_model(self.fc1, input_spike, fc1_u, fc1_v, fc1_s)
#             fc2_u, fc2_v, fc2_s = self.neuron_model(self.fc2, fc1_s, fc2_u, fc2_v, fc2_s)
#             fc3_u, fc3_v, fc3_s = self.neuron_model(self.fc3, fc2_s, fc3_u, fc3_v, fc3_s)
#             fc4_u, fc4_v, fc4_s = self.neuron_model(self.fc4, fc3_s, fc4_u, fc4_v, fc4_s)
#             fc4_sumspike += fc4_s
#         out = fc4_sumspike / self.batch_window
#         return out


# if __name__ == "__main__":

#     #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")
#     actor_net = ActorNetSpiking(22, 2, device)

#     data = [0.4,0.2,0.3,0.4,0.5,0.6,0.3,0.5,0.2,0.7,0.34,0.23,0.45,0.56,0.72,0.23,0.1,0.39,0.89,0.19,0.59,0.92]
#     data1 = np.array(data).reshape(1, -1)
#     data1 = data1.reshape((-1, 22, 1))
#     state_spikes = np.random.rand(3, 22, 10) < data1
#     state_spikes.astype(float)
#     spikes = torch.Tensor(state_spikes).to(device)
#     out = actor_net(spikes, 3).to(device)
#     print(out)


import torch
import torch.nn as nn
from typing import OrderedDict

class Trained_CNN(nn.Module):
    def __init__(self):
        super(Trained_CNN, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(1, 3, (3, 3), 1, 1, bias=True),  # 48 x 64 x 3
            nn.ReLU(),
            nn.Conv2d(3, 3, (2, 2), 2, 0, bias=True),  # 30 x 40 x 3
            nn.ReLU(),
            nn.Conv2d(3, 1, (2, 2), 2, 0, bias=True),  # 15 x 20 x 1
            nn.Flatten(),                              # 300
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net1(x)
        return out 



if __name__ == "__main__":

    device = torch.device('cuda')
    trained_CNN = Trained_CNN()
    
    file_name = 'DDPG_R1_cnn_network_s8.pt'
    state_dict = torch.load('./save_ddpg_weights/' + file_name)
    # print('========here1========')
    # for i ,j in state_dict.items():
    #     print(i)
    #     print(j)
    #     print()
    new_state_dict = OrderedDict()
    new_state_dict = {'net1.'+k:v for k,v in state_dict.items()}
    net_cnn_dict = trained_CNN.state_dict()
    # print('========here2========')
    # for i ,j in net_cnn_dict.items():
    #     print(i)
    #     print(j)
    #     print()
    net_cnn_dict.update(new_state_dict)
    trained_CNN.load_state_dict(net_cnn_dict)
    trained_CNN.to(device)
    print('here')