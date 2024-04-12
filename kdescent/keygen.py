import jax.random
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class KeyGenerator:
    def __init__(self, randkey=0):
        self.randkey = _init_randkey(randkey)

    @jax.jit
    def with_newkey(self):
        self.randkey = jax.random.split(self.randkey)[0]
        return self

    def tree_flatten(self):
        children = (self.randkey,)  # arrays / dynamic values
        aux_data = {}  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


def _init_randkey(randkey):
    if isinstance(randkey, int):
        randkey = jax.random.key(randkey)
    else:
        msg = f"Invalid {type(randkey)=}: Must be int or PRNG Key"
        assert hasattr(randkey, "dtype"), msg
        assert jnp.issubdtype(randkey.dtype, jax.dtypes.prng_key), msg

    return randkey
