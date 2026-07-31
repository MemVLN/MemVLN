import collections
import functools
import re

import torch
from typing import Any
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import PartialState
from accelerate.utils import DistributedType, recursively_apply
from accelerate.utils.constants import TORCH_DISTRIBUTED_OPERATION_TYPES

logger = get_logger(__name__)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def default_collate(batch):
    r"""
        Modify torch.utils.data.default_collate to support collating variable-length lists
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        try:
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem._typed_storage()._new_shared(numel, device=elem.device)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))
            return torch.stack(batch, 0, out=out)
        except:
            # custom: in case of inconsistent tensor shape
            return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: default_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        """custom part: directly return for lists and tuples"""
        return batch

    raise TypeError(default_collate_err_msg_format.format(elem_type))

# Customize operations for gathering
def _gpu_gather_object(object: Any):
    # by JY Huang: re-implement the method for gathering non-tensor objects
    output_objects = [None for _ in range(PartialState().num_processes)]
    torch.distributed.all_gather_object(output_objects, object)
    if isinstance(object, (list, tuple)):
        output_list = []
        for item in output_objects:
            output_list.extend(item)
        return output_list
    elif isinstance(object, dict):
        template = output_objects[0]
        output_dict = {}
        for k, v in template.items():
            if v is None or not hasattr(v, '__iter__'):
                output_dict[k] = v
                continue
            output_dict[k] = []
            for item in output_objects:
                output_dict[k].extend(item[k])
        return output_dict


def gather_object(object: Any):
    """
    Recursively gather object in a nested list/tuple/dictionary of objects from all devices.

    Args:
        object (nested list/tuple/dictionary of picklable object):
            The data to gather.

    Returns:
        The same data structure as `object` with all the objects sent to every device.
    """
    if PartialState().distributed_type == DistributedType.XLA:
        raise NotImplementedError("gather objects in TPU is not supported")
    elif PartialState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
        return _gpu_gather_object(object)
    else:
        return object


"""
Customize Accelerator to support:
    1. advanced gather_for_metrics
    2. only saving partial model weights when calling save_state
"""
class CustomAccelerator(Accelerator):

    def gather_for_metrics(self, input_data):
        # by JY Huang: re-implement this method for gathering non-tensor objects
        try:
            recursively_apply(lambda x: x, input_data, error_on_other_type=True)
            all_tensors = True
        except TypeError:
            all_tensors = False

        if not all_tensors:
            """ custom part 1 """
            data = gather_object(input_data)
            """ custom part 1 """
        else:
            data = self.gather(input_data)

        try:
            if self.gradient_state.end_of_dataloader:
                # at the end of a dataloader, `gather_for_metrics` regresses to
                # `gather` unless the dataset has a remainder so log.
                if self.gradient_state.remainder == -1:
                    logger.info(
                        "The used dataset had no length, returning gathered tensors. You should drop the remainder yourself."
                    )
                    return data
                elif self.gradient_state.remainder > 0:
                    """ custom part 2 """
                    # Last batch needs to be truncated on distributed systems as it contains additional samples
                    def _adjust_samples(tensor):
                        return tensor[: self.gradient_state.remainder] if tensor is not None else None
                    if all_tensors:
                        # This only applies to tensors, as defined in `recursively_apply`
                        return recursively_apply(_adjust_samples, data)
                    else:
                        if isinstance(data, (list, tuple)):
                            return _adjust_samples(data)
                        elif isinstance(data, dict):
                            return {k: _adjust_samples(v) for k, v in data.items()}
                        else:
                            raise NotImplementedError(f"Non-tensor gather only supports list, tuple or dict")
                    """ custom part 2 """
                else:  # remainder is 0
                    # no remainder even though at end of dataloader, so nothing to do.
                    return data
            else:
                # Not at the end of the dataloader, no need to adjust the tensors
                return data
        except Exception:
            # Dataset had no length or raised an error
            return data

    def get_state_dict(self, model, unwrap=True):
        state_dict = super().get_state_dict(model, unwrap=unwrap)
        # only save learnable parameters
        if not self.distributed_type == 'DEEPSPEED' or self.is_main_process:
            keys_list = list(state_dict.keys())
            for k in keys_list:
                if k not in self.learn_params_list:   # need to assign `learn_params_list` before calling this method
                    del state_dict[k]

        return state_dict


def dict_to_cuda(input_dict, device):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.to(device)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.to(device) for ele in v]
        # elif (
        #     isinstance(input_dict[k], list)
        #     and len(input_dict[k]) > 0
        #     and isinstance(input_dict[k][0], Det3DDataElement)
        # ):
        #     input_dict[k] = [ele.to(device) for ele in v]

    return input_dict
