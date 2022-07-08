import torch
import torch.nn as nn


class ActorNet(nn.Module):
    """ Actor Network """
    def __init__(self, state_num, action_num, hidden1=256, hidden2=256, hidden3=256, in_channel=1, mid_channel=3, out_channel=1):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(ActorNet, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, (3, 3), 1, 1, bias=True),   # 48 x 64 x 3
            nn.ReLU(),
            nn.Conv2d(mid_channel, mid_channel, (2, 2), 2, 0, bias=True),  # 30 x 40 x 3
            nn.ReLU(),
            nn.Conv2d(mid_channel, out_channel, (2, 2), 2, 0, bias=True),  # 15 x 20 x 1
            nn.Flatten(),
            nn.Sigmoid()                                                   # 300
        )

        self.net2 = nn.Sequential(
            nn.Linear(state_num + 192, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, action_num),
            nn.Sigmoid(),
        )

    def forward(self, xa):   # in:  depth_img and state(get from the /gazebo/get_model_states)
        depth_img, x = xa
        depth_img_out = self.net1(depth_img)          # [1, 300]
        tmp_data = torch.cat([depth_img_out, x], 1)   # [1, 300 + num_state]
        out = self.net2(tmp_data)
        return out


class CriticNet(nn.Module): 
    """ Critic Network"""
    def __init__(self, state_num, action_num, hidden1=512, hidden2=512, hidden3=512, in_channel=1, mid_channel=3, out_channel=1):
        """

        :param state_num: number of states
        :param action_num: number of actions
        :param hidden1: hidden layer 1 dimension
        :param hidden2: hidden layer 2 dimension
        :param hidden3: hidden layer 3 dimension
        """
        super(CriticNet, self).__init__()

        self.net1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, (3, 3), 1, 1, bias=True),   # 48 x 64 x 3
            nn.ReLU(),
            nn.Conv2d(mid_channel, mid_channel, (2, 2), 2, 0, bias=True),  # 24 x 32 x 3
            nn.ReLU(),
            nn.Conv2d(mid_channel, out_channel, (2, 2), 2, 0, bias=True),  # 12 x 16 x 1
            nn.Flatten(),                                                  # 192
            nn.Sigmoid()
        )

        self.net2 = nn.Sequential(
            nn.Linear(state_num + 192 + action_num, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, 1)
        )


    def forward(self, xa):  # in: depth, state(from /gazebo/get_model_state) and action

        x, depth_img, a = xa
        depth_img = self.net1(depth_img)
        out = self.net2(torch.cat([x, depth_img, a], 1))

        return out
