import time
from functools import wraps

import torch
from torch.utils.dlpack import (
    to_dlpack,
    from_dlpack
)
from tqdm import tqdm
import cupy as cp


def t2c(x):
    """
    Translate pytorch tensor into cuda array by dlpack which is a
    zero copy method.

    Parameters
    -----------
    x: pytorch tensor

    Returns
    -----------
    cx: cupy array
    """
    dx = to_dlpack(x)
    return cp.fromDlpack(dx)

def c2t(x):
    """
    Translate cupy array into pytorch tensor by dlpack

    Parameters
    -------------
    x: cupy array

    Returns
    -----------
    tx: pytorch tensor
    """
    dx = x.toDlpack()
    return from_dlpack(dx)

# TODO: update benchmark module
def benchmark(iters, warmup=3):
    def task_wrapper(task):
        @wraps(task)
        def benchmark_task(*args, **kwargs):
            # warmup
            print("====> warmup")
            for _ in range(warmup):
                task(*args, **kwargs)
            
            print("====> start test")
            y = None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in tqdm(range(iters)):
                y = task(*args, **kwargs)
            torch.cuda.synchronize()
            print((time.perf_counter() - t0) / iters)
            return y
        return benchmark_task
    return task_wrapper