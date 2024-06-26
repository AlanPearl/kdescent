from functools import partial

import tqdm
import numpy as np
import jax.random
import jax.numpy as jnp
import jaxopt
import optax

from . import keygen


def adam(lossfunc, init_params, n_iter=100, param_bounds=None,
         learning_rate=0.05, randkey=1, **other_kwargs):
    """
    Perform gradient descent

    Parameters
    ----------
    lossfunc : callable
        Function to be minimized via gradient descent. Must be compatible with
        jax.jit and jax.grad. Must have signature f(params, **other_kwargs)
    init_params : array-like
        Initial guess in parameter space
    n_iter : int, optional
        Number of gradient descent iterations to perform, by default 100
    param_bounds : Sequence, optional
        Lower and upper bounds of each parameter, by default None
    learning_rate : float, optional
        Initial Adam learning rate, by default 0.05
    randkey : int, optional
        Random seed or key, by default 1. If not None, lossfunc must accept
        the "randkey" keyword argument, e.g. `lossfunc(params, randkey=key)`

    Returns
    -------
    params : jnp.array
        List of params throughout the entire gradient descent, of shape
        (n_iter, n_param)
    state : OptaxState
        Contains information about the final state of the gradient descent
    """
    if param_bounds is None:
        return adam_unbounded(lossfunc, init_params, n_iter,
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
        ulossfunc, init_uparams, n_iter, learning_rate,
        randkey, **other_kwargs)
    params = apply_inverse_transforms(uparams.T, param_bounds).T

    return params, state


def adam_unbounded(lossfunc, init_params, n_iter=100,
                   learning_rate=1e-3, randkey=1, **other_kwargs):
    kwargs = {**other_kwargs}
    if randkey is not None:
        randkey = keygen.init_randkey(randkey)
        kwargs["randkey"] = randkey
    opt = optax.adam(learning_rate)
    solver = jaxopt.OptaxSolver(opt=opt, fun=lossfunc, maxiter=n_iter)
    state = solver.init_state(init_params, **kwargs)
    params = [init_params]
    for _ in tqdm.trange(n_iter):
        if randkey is not None:
            randkey = keygen.gen_new_key(randkey)
            kwargs["randkey"] = randkey
        params_i, state = solver.update(params[-1], state, **kwargs)
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
