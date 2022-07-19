# -*- coding: utf-8 -*-
"""
Created on 2020/8/17
@project: SPAIC
@filename: Network_saver
@author: Mengxiao Zhang
@contact: mxzhangice@gmail.com

@description:
对既定格式网络的存储
"""

import os
from ..Network.Assembly import Assembly
from ..Neuron.Neuron import NeuronGroup
from ..Neuron.Node import Node
from ..Network.Topology import Connection
from ..Backend.Backend import Backend
from ..Network.Topology import Projection
from ..Monitor.Monitor import Monitor

import time


def network_save(Net: Assembly, filename=None, path=None,
                 trans_format='json', combine=False, save=True, save_weight=True):
    '''
        Save network to files.

        Args:
            Net(Assembly) : The network needed to be saved.
            filename(Str) : The filename of the file that save target network.
            trans_format(str) : The format of file, could be 'json' or 'yaml'
            combine(Boolen) : Whether combine weight and structure of the Network into on file, False by default.
            save(Boolen) : Whether need to save the structure.

        Return:
            filename(str) : The filename of the file we save network, since it
                will give an auto name if no name given.

        Examples:
            >>> save_file = network_save(Net, "TestNetwork", trans_format='yaml', combine=True, save=True)
            ("TestNetwork", a dict of the structure of the network)
            or
            >>> save_file = network_save(Net, "TestNetwork", trans_format='json', combine=True, save=False)
            a dict of the structure of the network

    '''

    if filename is None:
        if Net.name:
            filename = Net.name + str(time.time())
        else:
            filename = "autoname" + str(time.time())

    origin_path = os.getcwd()
    if path:
        filedir = path + '/' + filename
    else:
        path = './'
        filedir = path + filename

    if save:
        os.chdir(path)
        if filename not in os.listdir():
            os.mkdir(filename)
        os.chdir(filedir)

    result_dict = trans_net(Net=Net, path=path, combine=combine, save=save, save_weight=save_weight)

    if save:
        if trans_format == "yaml":
            import yaml
            result = yaml.dump(result_dict, indent=4)
            ends = '.yml'
        elif trans_format == 'json':
            import json
            result = json.dumps(result_dict, indent=4)
            ends = '.json'
        else:
            raise ValueError("Wrong data format. Only support yaml/json format.")

        with open('./'+filename+ends, 'w+') as f:
            f.write(result)
        os.chdir(origin_path)
        print("Save Complete.")
        return filename
    else:
        os.chdir(origin_path)
        print("Complete.")
        return result_dict





def trans_net(Net: Assembly, path: str, combine: bool, save: bool, save_weight: bool):
    '''
        Transform the structure of the network for saving.

        Args:
            Net(Assembly): target network.
            path(string): Target path for saving net data.

        return:
            result_dict(dictionary) : the result diction of the whole Network.

        Example:
            yaml_net = trans_net(Net)

    '''
    result_dict = dict()
    net_name = Net.name
    result_dict[net_name] = []

    for g in Net._groups.values():
        if g._class_label == '<asb>':  # translate other assemblies
            sub_net_name = g.name
            result_dict[net_name].append(trans_net(g, path+'/'+str(sub_net_name),
                                                   combine, save, save_weight=False))
        elif g._class_label == '<neg>':  # translate layers
            result_dict[net_name].append(trans_layer(g))
        elif g._class_label == '<nod>':  # translate nodes
            result_dict[net_name].append(trans_node(g))
        else:
            # TODO: if get wrong _class_label, need check the type of
            #  this element
            pass

    for p in Net._projections.values():
        result_dict[net_name].append(trans_projection(p))

    for c in Net._connections.values():  # translate connections
        result_dict[net_name].append(trans_connection(c, combine, save_weight))

    if '_monitors' in dir(Net):
        mon_dict = {'monitor': []}
        result_dict[net_name].append(mon_dict)
        for monitor in Net._monitors.items():
            mon_dict['monitor'].append(trans_monitor(monitor))

    if '_learners' in dir(Net):
        for key, g in Net._learners.items():  # translate learners
            result_dict[net_name].append({key: trans_learner(g, key)})
    # result_dict[net_name].append({'learners':trans_learner(Net._learners)})

    if (not combine) and save_weight:
        if Net._backend:
            result_dict[net_name].append(
                {'backend': trans_backend(Net._backend, save)}
            )
        else:
            import warnings
            warnings.warn("Net._backend not exist. Please check whether need save weight")

    return result_dict


def trans_node(node: Node):
    '''
        Transform the structure of the Node layer for saving and extract the
            parameters.

        Args:
            node (Node): target node layer, like input layer and output layer

        return:
            result_dict (dictionary): the result diction with necessary
                parameters of the layer.

    '''

    needed = ['id', 'shape', 'num', '_time', '_dt', 'coding_method',
              'coding_var_name', 'type', 'name', 'coding_param']

    result_dict = dict()
    para_dict = dict()

    for key, para in node.__dict__.items():
        if key in needed:
            para_dict[key] = check_var_type(para)

    para_dict['shape'] = list(para_dict['shape'][1:])
    if node.is_encoded:
        para_dict['shape'] = para_dict['shape'][1:]

    if 'dt' in dir(node):
        para_dict['dt'] = node.dt
    if 'time' in dir(node):
        para_dict['time'] = node.time

    if node.__dict__['dec_target']:
        para_dict['dec_target'] = node.__dict__['dec_target'].name

    if 'action' in node.__dict__.keys():
        para_dict['kind'] = 'Action'
    elif 'reward' in node.__dict__.keys():
        para_dict['kind'] = 'Reward'
    elif 'predict' in node.__dict__.keys():
        para_dict['kind'] = 'Decoder'
    elif 'gen_first' in node.__dict__.keys():
        para_dict['kind'] = 'Generator'
    else:
        para_dict['kind'] = 'Encoder'

    para_dict['_class_label'] = '<nod>'
    result_dict[node.name] = para_dict
    return result_dict


def trans_layer(layer: NeuronGroup):
    '''
        Transform the structure of the layer for saving and extract the
            parameters.

        Args:
            layer (NeuronGroup): target layer

        return:
            result_dict (dictionary): the result diction with necessary
                parameters of the layer.

    '''
    result_dict = dict()
    para_dict = dict()


    unneeded = ['hided', '_backend', '_connections', '_supers', '_input_connections',
                '_output_connections', '_var_names', 'model_class', '_operations', 'model',
                '_groups']
    needed = ['model_name', 'id', 'name', 'num', 'position', 'shape', 'type', 'parameters']
    # Needed parameters: neuron_number, neuron_shape, neuron_type,
    # neuron_position, neuron_model, name, parameters.

    for key, para in layer.__dict__.items():
        if key in needed:
            para_dict[key] = check_var_type(para)

    if para_dict['position'] != ('x, y, z' or 'x, y'):
        para_dict.pop('position')
    para_dict['_class_label'] = '<neg>'

    para_dict['parameters'] = layer.parameters

    result_dict[layer.name] = para_dict
    return result_dict


def trans_projection(projection: Projection):
    '''
        Transform the structure of the projection for saving and extract the
            parameters.

        Args:
            projection (Projection): target projection

        return:
            result_dict (dictionary): the result diction with necessary
                parameters of the projection.

    '''
    result_dict = dict()
    para_dict = dict()
    name_needed = ['pre_assembly', 'post_assembly']
    needed = ['name', 'link_type', 'ConnectionParameters']

    for key, para in projection.__dict__.items():
        if key in name_needed:
            para_dict[key] = para.name
        elif key in needed:
            para_dict[key] = check_var_type(para)

    para_dict['_class_label'] = '<prj>'
    para_dict['_policies'] = []
    for ply in projection._policies:
        if ply.name == 'Index_policy':
            para_dict['_policies'].append(
                {'name': ply.name,
                 'pre_indexs': ply.pre_indexs,
                 'post_indexs': ply.post_indexs,
                 'level': ply.level}
            )
        else:
            para_dict['_policies'].append(
                {'name': ply.name,
                 'pre_types': list(ply.pre_types) if ply.pre_types else ply.pre_types,
                 'post_types': list(ply.post_types) if ply.post_types else ply.post_types,
                 'level': ply.level}
            )

    result_dict[projection.name] = para_dict

    return result_dict


def trans_connection(connection: Connection, combine: bool, save_weight: bool):
    '''
        Transform the structure of the connection for saving and extract the
            parameters.

        Args:
            connection (Connection): target connection
            combine (bool): whether combine weights.

        return:
            result_dict (dictionary): the result diction with necessary
                parameters of the connection.

    '''
    result_dict = dict()
    para_dict = dict()

    name_needed = ['pre_assembly', 'post_assembly']
    needed = ['name', 'link_type', 'synapse_type', 'max_delay', 'sparse_with_mask',
              'pre_var_name', 'post_var_name', 'parameters', 'id', ]
    unneeded = ['hided', 'pre_groups', 'post_groups', 'pre_assemblies', 'post_assemblies',
                'unit_connections', '_var_names', '_supers', '_backend']
    # **link_parameters

    for key, para in connection.__dict__.items():
        if key in name_needed:
            para_dict[key] = para.id
        elif key in needed:
            if key == 'parameters':
                if 'weight' in para.keys():
                    del para['weight']
            para_dict[key] = check_var_type(para)
    if combine:     # 是否需要在文件中存储weight
        para_dict['weight'] = check_var_type(connection.weight)

    para_dict['_class_label'] = '<con>'
    result_dict[connection.name] = para_dict

    return result_dict


def trans_backend(backend: Backend, save: bool):
    '''
    Transform the data of backend for saving.

    Args:
        backend: target backend.
        path(string): Target path for saving net data.

    Returns:
        result(dict): Contain the parameters of backend to be saved.
    '''

    # Needed parameters: _variables, _parameters_dict, _InitVariables_dict,
    # dt, time, time_step, _graph_var_dicts,

    # key_parameters_dict = ['_variables', '_parameters_dict', '_InitVariables_dict']
    key_parameters_dict = ['_parameters_dict']
    key_parameters_list = ['dt', 'runtime', 'time', 'n_time_step']


    if backend._variables is None:
        import warnings
        warnings.warn('Backend end don\'t have variables. Have not built Backend. Weight not exists.')
        return
    else:
        if 'parameters' not in os.listdir():
            os.mkdir('parameters')
    ori_path = os.getcwd()
    sim_path = ori_path + '/parameters'
    os.chdir(sim_path)

    import torch

    result_dict = dict()
    for key in key_parameters_dict:
        if save:
            save_path = sim_path + '/' + key + '.pt'
            data = backend.__dict__[key]
            torch.save(data, save_path)
            result_dict[key] = './parameters/' + key + '.pt'
        else:
            result_dict = backend._parameter_dict
            # raise ValueError("Wrong save choosen, since parameters can be get from network"
            #                  "unneeded to use network_save function.")

    for key in key_parameters_list:
        result_dict[key] = backend.__dict__[key]

    os.chdir(ori_path)
    return result_dict


def trans_learner(learner, learn_name):
    """
    Transform learner parameters to dict.
    Args:
        learner: Target learner with needed parameters.

    Returns:
        result(dict): Contain the parameters of learner to be saved.
    """
    import torch
    # result_dict = dict()
    para_dict = dict()
    trainables = ['trainable_connections', 'trainable_groups', 'trainable_nodes']
    para_dict['trainable'] = []
    needed = ['name', 'parameters', 'optim_name', 'optim_lr', 'optim_para', 'lr_schedule_name', 'lr_schedule_para']
    para_dict['_class_label'] = '<learner>'
    for key in needed:
        if key in learner.__dict__.keys():
            para = learner.__dict__.get(key)
            para_dict[key] = check_var_type(para)
                # if type(para) != torch.Tensor \
                #     else para.detach().cpu().numpy().tolist()

    for train_name in trainables:
        for key, train in learner.__dict__[train_name].items():
            para_dict['trainable'].append(check_var_type(train.name))

    para_dict['algorithm'] = check_var_type(para_dict['name'])
    para_dict['name'] = check_var_type(learn_name)
    if 'algorithm' in para_dict['parameters'].keys():
        del para_dict['parameters']['algorithm']

    return para_dict


def trans_monitor(monitor: Monitor):
    from spaic.Monitor.Monitor import StateMonitor, SpikeMonitor
    needed = ['var_name', 'index', 'dt', 'get_grad', 'nbatch']
    name, mon = monitor
    result_dict = dict()
    for i in needed:
        result_dict[i] = mon.__dict__[i]
    result_dict['target'] = mon.target.id
    result_dict['monitor_type'] = 'StateMonitor' if type(monitor[1]) == StateMonitor else 'SpikeMonitor'

    return {name: result_dict}


def check_var_type(var):
    import torch
    import numpy as np
    import json
    try:
        json.dumps(var)
        return var
    except:
        if isinstance(var, torch.Tensor):
            return var.detach().cpu().numpy().tolist()
        if isinstance(var, dict):
            for key, value in var.items():
                var[key] = check_var_type(value)
            return var
        try:
            var_list = var.tolist()
            return var_list
        except:
            raise TypeError('Please check type of parameters, we only support tensor or python build-in types.')
        # if len(var) >= 2:
        #     res_list = var.tolist()
        #     return [check_var_type(i) for i in res_list]
        # else:
        #     return check_var_type(var.tolist()[0])
    # if isinstance(var, torch.Tensor) or isinstance(var, np.ndarray):
    #     if len(var) >= 2:
    #         return var.tolist()
    #     else:
    #         return var.tolist()[0]
    # # elif isinstance(var, np.ndarray):
    # #     if len(var) >= 2:
    # #         return var.tolist()
    # #     else:
    # #         return var.tolist()[0]
    # elif isinstance(var, set):
    #     return list(var)
    # elif isinstance(var, list):  # 如果出现list中还有别的类型，考虑去后端解决
    #     return var
    # elif isinstance(var, dict):
    #     for key, value in var.items():
    #         var[key] = check_var_type(value)
    #     return var
    # else:
    #     return var



# def






