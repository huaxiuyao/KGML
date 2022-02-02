import torch
import os
import shutil
from typing import Tuple, List
import numpy as np
from tqdm import tqdm
import cv2
from torchvision.transforms import transforms
import pickle
import six
import ipdb

def mkdir(dir):
    """Create a directory, ignoring exceptions

    # Arguments:
        dir: Path of directory to create
    """
    try:
        os.mkdir(dir)
    except:
        pass


def rmdir(dir):
    """Recursively remove a directory and contents, ignoring exceptions

   # Arguments:
       dir: Path of directory to recursively remove
   """
    try:
        shutil.rmtree(dir)
    except:
        pass


def copy_weights(from_model: torch.nn.Module, to_model: torch.nn.Module):
    """Copies the weights from one model to another model.

    # Arguments:
        from_model: Model from which to source weights
        to_model: Model which will receive weights
    """
    if not from_model.__class__ == to_model.__class__:
        raise(ValueError("Models don't have the same architecture!"))

    for m_from, m_to in zip(from_model.modules(), to_model.modules()):
        is_linear = isinstance(m_to, torch.nn.Linear)
        is_conv = isinstance(m_to, torch.nn.Conv2d)
        is_bn = isinstance(m_to, torch.nn.BatchNorm2d)
        if is_linear or is_conv or is_bn:
            m_to.weight.data = m_from.weight.data.clone()
            if m_to.bias is not None:
                m_to.bias.data = m_from.bias.data.clone()


def autograd_graph(tensor: torch.Tensor) -> Tuple[
            List[torch.autograd.Function],
            List[Tuple[torch.autograd.Function, torch.autograd.Function]]
        ]:
    """Recursively retrieves the autograd graph for a particular tensor.

    # Arguments
        tensor: The Tensor to retrieve the autograd graph for

    # Returns
        nodes: List of torch.autograd.Functions that are the nodes of the autograd graph
        edges: List of (Function, Function) tuples that are the edges between the nodes of the autograd graph
    """
    nodes, edges = list(), list()

    def _add_nodes(tensor):
        if tensor not in nodes:
            nodes.append(tensor)

            if hasattr(tensor, 'next_functions'):
                for f in tensor.next_functions:
                    if f[0] is not None:
                        edges.append((f[0], tensor))
                        _add_nodes(f[0])

            if hasattr(tensor, 'saved_tensors'):
                for t in tensor.saved_tensors:
                    edges.append((t, tensor))
                    _add_nodes(t)

    _add_nodes(tensor.grad_fn)

    return nodes, edges


def compress(path, output):
    with np.load(path, mmap_mode="r") as data:
        images = data["images"]
        array = []
        for ii in tqdm(six.moves.xrange(images.shape[0]), desc='compress'):
            im = images[ii]
            im_str = cv2.imencode('.png', im)[1]
            array.append(im_str)
    with open(output, 'wb') as f:
        pickle.dump(array, f, protocol=pickle.HIGHEST_PROTOCOL)


def decompress(path, output):
    with open(output, 'rb') as f:
        array = pickle.load(f, encoding='bytes')
    images = torch.FloatTensor(len(array), 3, 84, 84)
    for ii, item in tqdm(enumerate(array), desc='decompress'):
        im = cv2.cvtColor(cv2.imdecode(item, 1), cv2.COLOR_BGR2RGB)
        images[ii] = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])(im)
    # np.savez(path, images=images)
    torch.save(images, path)

def get_L2feature_norm(x):
    l = (x.norm(p=2, dim=1).mean()) ** 2
    return l
