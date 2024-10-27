import os

import numpy as np

from tensordict.tensordict import TensorDict
from marpdan.problems import *
import torch
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.dirname(os.path.dirname(CURR_DIR))



def load_npz_to_tensordict(filename):
    """Load a npz file directly into a TensorDict
    We assume that the npz file contains a dictionary of numpy arrays
    This is at least an order of magnitude faster than pickle
    """
    x = np.load(filename)
    x_dict = dict(x)
    batch_size = x_dict[list(x_dict.keys())[0]].shape[0]
    return TensorDict(x_dict, batch_size=batch_size)

def load_pyth_to_tensordict(filename):
    data = torch.load(filename)
    x_dict = {}
    x_dict['locs'] = data.nodes[:,:,:2]
    x_dict['demand_linehaul']= data.nodes[:,1:,2]
    x_dict['time_windows'] = data.nodes[:,:,3:5]
    x_dict['service_time'] = data.nodes[:,:,5]
    x_dict['vehicle_capacity'] = torch.ones((data.nodes.size(0), 1))
    x_dict['speed'] = torch.full((data.nodes.size(0), 1), data.veh_speed)
    x_dict['dyn_time'] = data.nodes[:,:,-1]
    if type(data) == DVRPTW_QS_Dataset or type(data) == DVRPTW_QS_VB_Dataset:
        x_dict['changed_dem'] = data.changed_dem
    if type(data) == DVRPTW_VB_Dataset or type(data) == DVRPTW_QS_VB_Dataset:
        x_dict['b_time'] = data.b_time
        x_dict['b_veh_idx'] = data.b_veh_idx
    if type(data) == DVRPTW_TJ_Dataset or type(data) == DVRPTW_NR_TJ_Dataset:
        x_dict['distances_with_tj'] = data.distances_with_tj
        x_dict['tj_time'] = data.tj_time
    x_dict['problem_type'] = type(data).__name__
    batch_size = data.nodes.size(0)
    return TensorDict(x_dict, batch_size=batch_size)

def save_tensordict_to_npz(tensordict, filename, compress: bool = False):
    """Save a TensorDict to a npz file
    We assume that the TensorDict contains a dictionary of tensors
    """
    x_dict = {k: v.numpy() for k, v in tensordict.items()}
    if compress:
        np.savez_compressed(filename, **x_dict)
    else:
        np.savez(filename, **x_dict)


def check_extension(filename, extension=".npz"):
    """Check that filename has extension, otherwise add it"""
    if os.path.splitext(filename)[1] != extension:
        return filename + extension
    return filename


def load_solomon_instance(name, path=None, edge_weights=False):
    """Load solomon instance from a file"""
    import vrplib

    if not path:
        path = "data/solomon/instances/"
        path = os.path.join(ROOT_PATH, path)
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = f"{path}{name}.txt"
    if not os.path.isfile(file_path):
        vrplib.download_instance(name=name, path=path)
    return vrplib.read_instance(
        path=file_path,
        instance_format="solomon",
        compute_edge_weights=edge_weights,
    )


def load_solomon_solution(name, path=None):
    """Load solomon solution from a file"""
    import vrplib

    if not path:
        path = "data/solomon/solutions/"
        path = os.path.join(ROOT_PATH, path)
    if not os.path.isdir(path):
        os.makedirs(path)
    file_path = f"{path}{name}.sol"
    if not os.path.isfile(file_path):
        vrplib.download_solution(name=name, path=path)
    return vrplib.read_solution(path=file_path)
