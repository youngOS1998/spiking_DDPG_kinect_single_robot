# -*- coding: utf-8 -*-
"""
Created on 2020/9/14
@project: SPAIC
@filename: Torch_Backend
@author: Hong Chaofei
@contact: hongchf@gmail.com
@description:
"""
from .Backend import Backend, backends
import torch
import numpy as np
# from torch import fx
#from torch.nn import Module, Parameter
import torch.nn.functional as fn
import torch.nn as nn
from typing import Tuple, Dict, Callable

class Torch_Engine(torch.nn.Module):
    def __init__(self, graph_operations):
        super(Torch_Engine, self).__init__()
        self._graph_operations = graph_operations

    def forward(self, variables: Dict[str,torch.Tensor]):
        temp_dict = dict()
        update_dict = dict()
        reduce_dict = dict()

        for op in self._graph_operations:
            # for inputs
            inputs = []
            for var in op[2]:
                if var[0] == 'variables_dict':
                    inputs.append(variables[var[1]])
                elif var[0] == 'temp_dict':
                    inputs.append(temp_dict[var[1]])
                elif var[0] == 'update_dict':
                    inputs.append(update_dict[var[1]])
                elif var[0] == 'reduce_dict':
                    inputs.append(reduce_dict[var[1]])
            # compute the operation
            result = op[1](*inputs)
            if len(op[0]) == 1: result = [result]
            # assign the result variables
            for ind, var in enumerate(op[0]):
                if var[0] == 'temp_dict':
                    temp_dict[var[1]] = result[ind]
                elif var[0] == 'update_dict':
                    update_dict[var[1]] = result[ind]
                elif var[0] == 'reduce_dict':
                    if var[1] in reduce_dict:
                        reduce_dict[var[1]].append(result[ind])
                    else:
                        reduce_dict[var[1]] = [result[ind]]

        return update_dict



class Torch_Backend(Backend):
    backend_name = 'pytorch'

    def __init__(self, device='cpu'):
        super(Torch_Backend, self).__init__()

        self.device = device
        self.data_type = torch.float
        self.debug_data = []
        pass

    def build(self):
        from torch import fx
        # self._graph_var_dicts = {'variables_dict': self._variables, 'temp_dict': dict(), 'update_dict': dict(),
        #                          'reduce_dict': dict()}
        # self._graph_var_dicts['temp_dict']['example_temp_dict_pytorch_datatype'] = torch.empty(1)
        # self._graph_var_dicts['update_dict']['example_temp_dict_pytorch_datatype'] = torch.empty(1)
        # self._graph_var_dicts['reduce_dict']['example_temp_dict_pytorch_datatype'] = torch.empty(1)
        #
        #self.update_step = jit.trace(self.update_step)
        self.engine = Torch_Engine(self._graph_operations)
        self.engine = fx.symbolic_trace(self.engine)

        # self.graph_update_step = torch.jit.script(self.engine)
        self.graph_update_step = self.engine
        # print(self.engine.code)



    # def graph_update_step(self):
    #
    #     for op in self._graph_operations:
    #         inputs = []
    #         for var in op[2]:
    #             inputs.append(self._graph_var_dicts[var[0]][var[1]])
    #
    #         if op[0][0] is 'reduce_dict':
    #             self._graph_var_dicts['reduce_dict'][op[0][1]].append(op[1](*inputs))
    #         else:
    #             self._graph_var_dicts[op[0][0]][op[0][1]] = op[1](*inputs)
    #
    #     return tuple(self._graph_var_dicts['variables_dict'].values())

    #  As of now, autograd support floating point Tensor types ( half, float, double and bfloat16) and complex Tensor types (cfloat, cdouble).
    def add_backend_variable(self, name, shape, value=None, grad=False, is_sparse=False, init=None, init_param=None):
        '''
        Parameters
        ----------
        name
        shape
        value
        init
        Returns
        -------
        '''

        if init_param is None:
            init_param = dict()
        if value is not None:
            if hasattr(value, "__len__"):
                if tuple(value.shape) != tuple(shape):
                    raise ValueError("Value is not scalar and the shape of Value is not equal to shape")
                # add a sparse matrices with all dimensions greater than 2
                if is_sparse:
                    i = np.nonzero(value)
                    v = value[i]

                    # Index for sparse matrix
                    sparse_index = name + '_sparse_index'
                    self._variables[sparse_index] = torch.LongTensor(i).to(device=self.device)
                    self._InitVariables_dict[sparse_index] = self._variables[sparse_index]

                    # Value for sparse matrix
                    sparse_value = name + '_sparse_value'
                    if init is not None:
                        # self._variables[sparse_value] = self.init_param(True, init)
                        data = torch.empty(shape, dtype=torch.float32, device=self.device, requires_grad=True)
                        self._variables[sparse_value] = self.param_init_operate[init](data, *init_param)
                    else:
                        self._variables[sparse_value] = torch.tensor(v, dtype=torch.float32, requires_grad=True, device=self.device)
                    self._parameters_dict[sparse_value] = self._variables[sparse_value]

                    # The shape of sparse matrix
                    sparse_shape = name + '_sparse_shape'
                    self._variables[sparse_shape] = torch.Size(shape)
                    self._InitVariables_dict[sparse_shape] = self._variables[sparse_shape]

                    # Sparse matrix
                    self._variables[name] = torch.sparse.FloatTensor(self._variables[sparse_index], self._variables[sparse_value], self._variables[sparse_shape])
                else:
                    # add a non sparse matrices with all dimensions greater than 2
                    if init is not None:
                        data = torch.empty(shape, dtype=torch.float32, device=self.device, requires_grad=grad)
                        init = init.lower()
                        if init in self.param_init_operate.keys():
                            self._variables[name] = self.param_init_operate[init](data, *init_param)
                        else:
                            raise ValueError("No initialize method: %s in param_init_operate" % init)
                    else:
                        if isinstance(value, torch.Tensor):
                            self._variables[name] = value.clone().detach()
                        else:
                            self._variables[name] = torch.tensor(value, dtype=torch.float32, device=self.device,
                                                                 requires_grad=grad)

            elif len(shape) == 0:
                # add constant
                self._variables[name] = torch.tensor(value, dtype=torch.float32, device=self.device, requires_grad=grad)

            else:
                # add a matrix through constant
                if init is not None:
                    # self._variables[name] = self.init_param(grad, init)
                    data = value*torch.ones(shape, dtype=torch.float32, device=self.device, requires_grad=grad)
                    init = init.lower()
                    if init in self.param_init_operate.keys():
                        self._variables[name] = self.param_init_operate[init](data, *init_param)
                    else:
                        raise ValueError("No initialize method: %s in param_init_operate" % init)
                else:
                    self._variables[name] = value*torch.ones(shape, dtype=torch.float32, device=self.device, requires_grad=grad)

        return self._variables[name]

    def set_variable_value(self, name, value, is_parameter):
        if is_parameter:
            assert name in self._parameters_dict
            assert self._parameters_dict[name].shape == value.shape
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=self._parameters_dict[name].dtype, device=self._parameters_dict[name].device)
            with torch.no_grad():
                self._parameters_dict[name].data = value
        else:
            assert name in self._InitVariables_dict
            self._backend._InitVariables_dict[name].shape == value.shape
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=self._InitVariables_dict[name].dtype, device=self._InitVariables_dict[name].device)
            with torch.no_grad():
                self._InitVariables_dict[name].data = value

    # def init_param(self, grad, *init):
    #     if init[0] in self.param_init_operate:
    #         init_op = self.param_init_operate[init[0]]
    #     else:
    #         raise ValueError("No init operate %s in param_init_operate" % init[0])
    #     inputs = []
    #     shape = init[1]
    #     data = torch.empty(shape, dtype=torch.float32, device=self.device, requires_grad=grad)
    #     inputs.append(data)
    #
    #     for var in init[2:]:
    #         inputs.append(var)
    #     return init_op(*inputs)

    def sparse_to_dense(self, index_name, value_name, shape_name):
        return torch.sparse.FloatTensor(self._variables[index_name], self._variables[value_name], self._variables[shape_name])

    def get_str(self, level):
        return level*' ' + 'torch_backend'

    def threshold(self, x, v_th):
        return torch.gt(x, v_th).float()

    def cat(self, x, dim=1):
        return torch.cat(x, dim)

    def stack(self, x, dim=1):       # 在指定维度dim上连接（concatenate）若干个张量。
        try:
            return torch.stack(x, dim)
        except:
            # patch for SLIF 2[O]
            for ii in range(len(x)):
                if x[ii].dim() ==2:
                    tmp = torch.zeros_like(x[ii])
                    tmp = torch.stack([x[ii], tmp], dim=1)
                    x[ii] = tmp
            return torch.stack(x, dim)

    def reduce_sum(self, x, *dim):
        if len(dim)==0:
            dim = 1
        return torch.sum(x, dim=dim)

    def index_select(self, x,  indices, dim=1):
        return torch.index_select(x, dim=dim, index=indices)

    def permute(self, x, permute_dim):
        return x.permute(permute_dim)

    def view(self, x, view_dim):

        x = x.contiguous().view(view_dim)
        return x


    def scatter(self, x, indices):
        return torch.scatter(x, dim=0, index=indices)

    def conv1d(self, x, kernel):
        return torch.conv1d(x, kernel)

    def conv_trans1d(self, x, kernel):
        return torch.conv_transpose1d(x, kernel)

    def conv_2d(self, x, kernel, stride, padding, dilation, groups):
        if x.dim() == kernel.dim() + 1:
            xshape = list(x.shape)
            xshape[0] = xshape[0]*xshape[1]
            extend_size = xshape[1]
            xshape.pop(1)
            out = fn.conv2d(x.reshape(xshape), kernel, stride=stride, padding=padding, dilation=dilation, groups=groups)
            outshape = list(out.shape)
            outshape[0] = outshape[0]//extend_size
            outshape.insert(1, extend_size)
            return out.view(outshape)
        else:
            return fn.conv2d(x, kernel, stride=stride, padding=padding, dilation=dilation, groups=groups)

    def conv_max_pool2d(self, x, kernel, pool_kernel, stride, pool_stride, padding, pool_padding, dilation, groups):
        return fn.max_pool2d(fn.conv2d(x, kernel, stride=stride, padding=padding,
                             dilation=dilation, groups=groups), kernel_size=pool_kernel,
                             stride=pool_stride, padding=pool_padding)

        # return fn.conv2d(fn.max_pool2d(x, int(max_kernel[0])), kernel, stride=int(stride), padding=int(padding), dilation=int(dilation), groups=int(groups))

    def conv_avg_pool2d(self, x, kernel, pool_kernel, stride, pool_stride, padding, pool_padding, dilation, groups):
        return fn.avg_pool2d(fn.conv2d(x, kernel, stride=stride, padding=padding,
                             dilation=dilation, groups=groups), kernel_size=pool_kernel,
                             stride=pool_stride, padding=pool_padding)
    def conv_add_bias(self, x, bias):
        bias_t = bias.repeat(x.shape[-2], x.shape[-1], 1).permute(2, 1, 0)
        return x+bias_t
    def max_pool2d(self, x, pool_kernel, pool_stride, pool_padding):
        return fn.max_pool2d(x, kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)

    def avg_pool2d(self, x, pool_kernel, pool_stride, pool_padding):
        return fn.avg_pool2d(x, kernel_size=pool_kernel, stride=pool_stride, padding=pool_padding)

    def dropout(self, x, p, inplace=False):
        return fn.dropout(x, p=p, inplace=inplace)

    def reshape_mat_mult(self, A, X):

        if A.dim() == 4:
            (batchsize, outchannels, H, W) = A.shape
            A = A.view(batchsize, -1)
        elif A.dim() == 5:
            (batchsize, extend, outchannels, H, W) = A.shape
            A = A.view(batchsize, extend, -1)

        return torch.matmul(A, X.permute(1, 0))

    def im2col_indices(self, x, kh, kw, padding, stride):
        return fn.unfold(x, (kh, kw), padding=padding, stride=stride)

    def conv2d_flatten(self, x):
        return x.view(x.shape[0], x.shape[1], -1)

    def feature_map_flatten(self, x):
        return x.view(x.shape[0], -1)

    def add(self, x, y):

        return x + y

    def minus(self, x, y):
        return x - y

    def div(self, x, y):
        return torch.div(x, y)

    def relu(self, x):
        return torch.relu(x)

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def mat_mult_weight(self, A, X):
        '''
        Parameters
        ----------
        A--->preGroup:input
        X--->postGroup:weight
        Returns
        -------
        '''
        X = X.permute(1, 0)
        return torch.matmul(A, X)


    def mat_mult_pre(self, A, X):
        '''
        Parameters
        ----------
        A--->preGroup:input
        X--->postGroup:weight
        Returns
        -------
        '''
        A = A.permute(1, 0)
        return torch.matmul(A, X)

    def mat_mult(self, A, X):
        '''
        Parameters
        ----------
        A--->preGroup:input
        X--->postGroup:weight
        Returns
        -------
        '''
        return torch.matmul(A, X)

    def bmm(self, A, X):
        '''
        Parameters
        ----------
        A---> postGroup
        X---> preGroup
        Returns
        -------
        '''
        return torch.bmm(A, X)

    def ger(self, A, X):
        '''
        Parameters
        ----------
        A---> postGroup
        X---> preGroup
        Returns
        -------
        '''
        return torch.ger(A, X)

    def sparse_mat_mult_weight(self, A, X):
        '''
       Parameters
       ----------
       A--->preGroup:sparseWeight(post, pre)
       X--->postGroup:input(batch, pre)
       Returns
       -------
       '''
        X = X.permute(1, 0)
        result = torch.sparse.mm(A, X)
        result = result.permute(1, 0)
        return result

    def var_mult(self, A, X):
        return A * X

    def mult_sum_weight(self, A, X):
        X = X.permute(1, 0)
        A = A.permute(0, 2, 1)
        return torch.sum(torch.matmul(A, X), dim=-2)

    def mat_linear(self, A, X, b):
        return torch.matmul(A, X) + b

    def var_linear(self, A, X, b):

        return A*X+b

    def unsqueeze(self, X, dim):
        return torch.unsqueeze(X, dim)

    def to_numpy(self, data: torch.Tensor):
        return data.detach().cpu().numpy()

    def to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(torch.float).to(self.device)
        else:
            return torch.tensor(data, dtype=torch.float, device=self.device)

    def exp(self, x):
        return torch.exp(x)


    def clamp_(self, data, min, max):
        with torch.no_grad():
            data.clamp_(min, max)


    def clamp_max_(self, data, max):
        with torch.no_grad():
            data.clamp_max_(max)



    def clamp_min_(self, data, min):
        with torch.no_grad():
            data.clamp_min_(min)


    def uniform(self, data, a=-0.0, b=1.0):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            a(float): the lower bound of the uniform distribution
            b(float): the upper bound of the uniform distribution
        Returns:
            torch.nn.init.uniform_(data, a=0.0, b=1.0)
        '''
        return torch.nn.init.uniform_(data, a, b)


    def normal(self, data, mean=0.0, std=1.0):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            mean(float): the mean of the normal distribution
            std(float): the standard deviation of the normal distribution
        Returns:
            torch.nn.init.normal_(data, mean=0.0, std=1.0)
        '''
        return torch.nn.init.normal_(data, mean, std)



    def xavier_normal(self, data, gain=1.0):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            gain: an optional scaling factor
        Returns:
            torch.nn.init.xavier_normal_(data, gain=1.0)
        '''
        return torch.nn.init.xavier_normal_(data, gain)


    def xavier_uniform(self, data, gain=1.0):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            gain: an optional scaling factor
        Returns:
            torch.nn.init.xavier_uniform_(data, gain=1.0)
        '''
        return torch.nn.init.xavier_uniform_(data, gain)


    def kaiming_normal(self, data, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            a: the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
            mode: either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
            nonlinearity: the non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu' (default).
        Returns:
            torch.nn.init.kaiming_normal_(data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        '''
        return torch.nn.init.kaiming_normal_(data, a, mode, nonlinearity)


    def kaiming_uniform(self, data, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            a: the negative slope of the rectifier used after this layer (only used with 'leaky_relu')
            mode: either 'fan_in' (default) or 'fan_out'. Choosing 'fan_in' preserves the magnitude of the variance of the weights in the forward pass. Choosing 'fan_out' preserves the magnitudes in the backwards pass.
            nonlinearity: the non-linear function (nn.functional name), recommended to use only with 'relu' or 'leaky_relu' (default).
        Returns:
            torch.nn.init.kaiming_uniform_(data, a=0, mode='fan_in', nonlinearity='leaky_relu')
        '''
        return torch.nn.init.kaiming_uniform_(data, a, mode, nonlinearity)

    def zero_init(self, data, constant_value=0.0):
        '''
        Args:
            data(tensor): an n-dimensional torch.Tensor
            constant_value(float): the value to fill the tensor with
        Returns:
            torch.nn.init.constant_(data, constant_value)
        '''
        return torch.nn.init.constant_(data, constant_value)

    def sin(self, x):
        return torch.sin(x)

    def cos(self, x):
        return torch.cos(x)

    def tan(self, x):
        return torch.tan(x)

    def log(self, x):
        return torch.log(x)

    def log2(self, x):
        return torch.log2(x)

    def log10(self, x):
        return torch.log10(x)

    # def reset(self, x, v_reset, u_reset, spike):
    #
    #     # if hasattr(x, "__len__"):
    #     #     if x.shape != spike.shape:
    #     #         raise ValueError("%s and %s do not match" % (x.shape, spike.shape))
    #     mask = torch.eq(spike, 1)
    #     x[mask] = v_reset
    #     x[mask] += u_reset
    #     return x

    # def izh_v(self, v, u, psp):
    #     v = v+self.dt*(0.04*v*v+5*v+140-u+psp)
    #     return v
    #
    # def izh_u(self, a, b, v, u):
    #     u = u+self.dt*a*(b*v-u)
    #     return u


backends[Torch_Backend.backend_name] = Torch_Backend





# test = Torch_Backend()
# th = test.basic_operate['threshold']
# print(th(-1.0))