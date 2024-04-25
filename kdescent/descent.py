from functools import partial

import tqdm
import numpy as np
import jax.random
import jax.numpy as jnp
import jaxopt
import optax

from .keygen import KeyGenerator


def adam(lossfunc, init_params, maxiter=100, param_bounds=None,
         learning_rate=0.05, randkey=1, **other_kwargs):
    if param_bounds is None:
        return adam_unbounded(lossfunc, init_params, maxiter,
                              learning_rate, randkey, **other_kwargs)

    assert len(init_params) == len(param_bounds)
    if hasattr(param_bounds, "tolist"):
        param_bounds = param_bounds.tolist()
    param_bounds = [b if b is None else tuple(b) for b in param_bounds]

    def ulossfunc(uparams, *args, **kwargs):
        params = apply_inverse_transforms(uparams, param_bounds)
        return lossfunc(params, *args, **kwargs)

    init_uparams = apply_transforms(init_params, param_bounds)
    uparams, state = adam_unbounded(
        ulossfunc, init_uparams, maxiter, learning_rate,
        randkey, **other_kwargs)
    params = apply_inverse_transforms(uparams.T, param_bounds).T

    return params, state


def adam_unbounded(lossfunc, init_params, maxiter=100,
                   learning_rate=1e-3, randkey=1, **other_kwargs):
    lossfunc_kwargs = {**other_kwargs}
    if randkey is not None:
        randkey = KeyGenerator(randkey)
        lossfunc_kwargs["randkey"] = randkey.randkey
    opt = optax.adam(learning_rate)
    solver = jaxopt.OptaxSolver(
        opt=opt, fun=lossfunc, maxiter=maxiter)
    state = solver.init_state(init_params, **lossfunc_kwargs)
    params = [init_params]
    for _ in tqdm.trange(maxiter):
        if randkey is not None:
            randkey = randkey.with_newkey()
            lossfunc_kwargs["randkey"] = randkey.randkey
        params_i, state = solver.update(
            params[-1], state, **lossfunc_kwargs)
        params.append(params_i)

    return jnp.array(params), state


def apply_transforms(params, bounds):
    return jnp.array([transform(param, bound)
                      for param, bound in zip(params, bounds)])


def apply_inverse_transforms(uparams, bounds):
    return jnp.array([inverse_transform(uparam, bound)
                      for uparam, bound in zip(uparams, bounds)])


@partial(jax.jit, static_argnums=[1])
def transform(param, bounds):
    """Transform param into unbound param"""
    if bounds is None:
        return param
    low, high = bounds
    low_is_finite = low is not None and np.isfinite(low)
    high_is_finite = high is not None and np.isfinite(high)
    if low_is_finite and high_is_finite:
        mid = (high + low) / 2.0
        scale = (high - low) / jnp.pi
        return scale * jnp.tan((param - mid) / scale)
    elif low_is_finite:
        return param - low + 1.0 / (low - param)
    elif high_is_finite:
        return param - high + 1.0 / (high - param)
    else:
        return param


@partial(jax.jit, static_argnums=[1])
def inverse_transform(uparam, bounds):
    """Transform unbound param back into param"""
    if bounds is None:
        return uparam
    low, high = bounds
    low_is_finite = low is not None and np.isfinite(low)
    high_is_finite = high is not None and np.isfinite(high)
    if low_is_finite and high_is_finite:
        mid = (high + low) / 2.0
        scale = (high - low) / jnp.pi
        return mid + scale * jnp.arctan(uparam / scale)
    elif low_is_finite:
        return 0.5 * (2.0 * low + uparam + jnp.sqrt(uparam**2 + 4))
    elif high_is_finite:
        return 0.5 * (2.0 * high + uparam - jnp.sqrt(uparam**2 + 4))
    else:
        return uparam
