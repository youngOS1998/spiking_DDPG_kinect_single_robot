import os

os.chdir("../../")
import spaic
# import torch.jit as jit
import numpy as np
from spaic.Learning.STCA_Learner import STCA
from spaic.Learning.Learner import Learner
from tqdm import tqdm
from spaic.Network import Network
import torch
import torch.nn.functional as F
from spaic.IO.Dataset import MNIST as dataset



# 设备设置
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'
backend = spaic.Torch_Backend(device)
sim_name = backend.backend_name
sim_name = sim_name.lower()
backend.dt =0.1
print(device)

# 创建训练数据集
root = './spaic/Datasets/MNIST'
train_set = dataset(root, is_train=True)
test_set = dataset(root, is_train=False)
run_time = 20.0
node_num = dataset.maxNum
label_num = dataset.class_number
bat_size = 100
# 创建DataLoader迭代器
train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=False, drop_last=False)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)


class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()


        # can set neuron param dict, including 'tau_p', 'tau_q', 'tau_m', 'v_th' and 'v_reset'
        self.input = spaic.Encoder(shape=(1, 28, 28), num=node_num, coding_time=run_time,
                                     coding_method='poisson')#需要将input_channel也传进去
        # can set neuron param dict, including 'tau_p', 'tau_q', 'tau_m', 'v_th' and 'v_reset'

        #self.layer1 = spaic.NeuronGroup(676, neuron_shape=None, neuron_model='clif', batch_size=bat_size)  #26*26*4 经过池化 13*13*4, kernel_size=2
        self.layer1 = spaic.NeuronGroup(4 * 26 * 26, neuron_shape=None, neuron_model='lif',
                                          batch_size=bat_size)  #24*24*4经过池化 12*12*4, kernel_size=2


        #self.layer2 = spaic.NeuronGroup(200, neuron_shape=None, neuron_model='clif',batch_size=bat_size)  # 11*11*8经过池化 5*5*8, kernel_size=2
        self.layer2 = spaic.NeuronGroup(8 * 24 * 24, neuron_shape=None, neuron_model='lif',
                                          batch_size=bat_size)  # 8*8*8经过池化 4*4*8, kernel_size=2


        self.layer3 = spaic.NeuronGroup(label_num, neuron_model='lif')
        self.output = spaic.Decoder(num=label_num, dec_target=self.layer3,
                                      coding_time=run_time, coding_method='spike_counts')

        # Connection
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='conv', in_channels=1, out_channels=4,
                                              kernel_size=(3, 3), # maxpool_on=True, maxpool_kernel_size=(2, 2),
                                              w_std=0.05, w_mean=0.05)


        self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='conv', in_channels=4, out_channels=8,
                                              kernel_size=(3, 3), # maxpool_on=False, maxpool_kernel_size=(2, 2),
                                              w_std=0.05, w_mean=0.05)

        self.connection3 = spaic.Connection(self.layer2, self.layer3, link_type='full',
                                            syn_type=['flatten', 'basic_synapse'],
                                            w_std=0.05, w_mean=0.02)
        # Learner
        self._learner = Learner(algorithm='STCA', a=0.01, trainable=[self.connection1, self.connection2, self.connection3, self.layer1, self.layer2, self.layer3])
        self.set_backend(backend)
        # Minitor
        # self.mon_O = spaic.StateMonitor(self.input, 'O')


Net = TestNet()

Net.build(backend)  # 创建网络

param = Net.get_testparams()  # 得到网络模型参数

optim = torch.optim.Adam(param, lr=0.01)  # 创建优化器对象，并传入网络模型的参数
shedule = torch.optim.lr_scheduler.StepLR(optim, 5)
print("Start running")

eval_losses = []
eval_acces = []
losses = []
acces = []
# with torch.autograd.set_detect_anomaly(True):
for epoch in range(1):
        # 训练阶段
        pbar = tqdm(total=len(train_loader))
        train_loss = 0
        train_acc = 0

        for i, item in enumerate(train_loader):
            # 前向传播

            data, label = item
            if Net.connection1.link_type is 'conv':

                data = data.reshape(train_loader.batch_size, Net.input.shape[-3], Net.input.shape[-2], Net.input.shape[-1])#input.shape[0]:H, input.shape[1]:W


            Net.input(data)
            Net.output(label)
            Net.run(run_time)

            output = Net.output.predict

            if sim_name == 'pytorch':
                label = torch.tensor(label, device=device, dtype=torch.long)#.unsqueeze(dim=1)


            batch_loss = F.cross_entropy(output, label)
            optim.zero_grad()
            batch_loss.backward(retain_graph=False)
            optim.step()


            train_loss += batch_loss.item()
            predict_labels = torch.argmax(output, 1)
            num_correct = (predict_labels == label.view(-1)).sum().item()  # 记录标签正确的个数
            acc = num_correct / data.shape[0]
            train_acc += acc
            # out1 = np.mean(np.sum(Net.layer1_O.values[0,...], axis=-1))


            pbar.set_description_str(
                "[loss:%f, acc:%f]Batch progress: " % (batch_loss.item(), acc))
            pbar.update()
            # break
        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))
        print('train_acc', train_acc / len(train_loader))
        pbar.close()
        #
        #
        # # 测试阶段
        # eval_loss = 0
        # eval_acc = 0
        # pbarTest = tqdm(total=len(test_loader))
        # with torch.no_grad():
        #     for i, item in enumerate(test_loader):
        #         data, label = item
        #         if Net.connection1.link_type is 'conv':
        #
        #             data = data.reshape(train_loader.batch_size, Net.input.shape[-3], Net.input.shape[-2], Net.input.shape[-1])#input.shape[0]:H, input.shape[1]:W
        #
        #         Net.input(data)
        #         Net.run(run_time)
        #         output = Net.output.predict
        #         #output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
        #         if sim_name == 'pytorch':
        #             label = torch.tensor(label, device=device, dtype=torch.long)
        #         # q = torch.gather(output, dim=1, index=label.unsqueeze(dim=1).type(torch.int64))
        #         # batch_loss = torch.mean(-torch.log(q + 1.0e-8))
        #         batch_loss = F.cross_entropy(output, label)
        #         eval_loss += batch_loss.item()
        #
        #         _, pred = output.max(1)
        #         num_correct = (pred == label).sum().item()
        #         acc = num_correct / data.shape[0]
        #         eval_acc += acc
        #         pbarTest.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
        #         pbarTest.update()
        #     eval_losses.append(eval_loss / len(test_loader))
        #     eval_acces.append(eval_acc / len(test_loader))
        # pbarTest.close()
        # print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f},Test Loss:{:.4f},Test Acc:{:.4f}'
        #       .format(epoch, train_loss / len(train_loader), train_acc / len(train_loader),
        #               eval_loss / len(test_loader), eval_acc / len(test_loader)))
        # print("")
        #

from spaic.Library.Network_saver import network_save

from spaic.Library.Network_loader import network_load

test_str, test_data = network_save(Net, 'TestNet', 'yaml', False, True)

# test_2 = network_load(test_str)
# print('t')
