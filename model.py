import functools
import json
import math
from typing import Tuple, TypeVar
import warnings

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
import plotnine as gg
import matplotlib.pyplot as plt

# Type definitions for clarity
from typing import TypeVar, Tuple
T = TypeVar('T')
Pair = Tuple[T, T]

# Neural network definition
def unroll_net(seqs: jnp.ndarray):
    core = hk.LSTM(128)
    batch_size = seqs.shape[1]
    outs, state = hk.dynamic_unroll(core, seqs, core.initial_state(batch_size))
    return hk.BatchApply(hk.Linear(6))(outs), state

# Initialize model
model = hk.transform(unroll_net)
