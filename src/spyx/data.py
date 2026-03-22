import jax
import jax.numpy as jnp


def shift_augment(max_shift=10, axes=(-1,)):
    """Shift data augmentation tool. Rolls data along specified axes randomly up to a certain amount.

        
    :max_shift: maximum to which values can be shifted.
    :axes: the data axis or axes along which the input will be randomly shifted.
    """

    def _shift(data, rng):
        shift = jax.random.randint(rng, (len(axes),), -max_shift, max_shift)
        return jnp.roll(data, shift, axes)
    
    return jax.jit(_shift)


def shuffler(dataset, batch_size):
    """
    Higher-order-function which builds a shuffle function for a dataset.

    :dataset: jnp.array [# samples, time, channels...]
    :batch_size: desired batch size.
    """
    x, y = dataset
    cutoff = (y.shape[0] // batch_size) * batch_size
    data_shape = (-1, batch_size) + x.shape[1:]

    def _shuffle(dataset, shuffle_rng):
        """
        Given a dataset as a single tensor, shuffle its batches.

        :dataset: tuple of jnp.arrays with shape [# batches, batch size, time, ...] and [# batches, batchsize]
        :shuffle_rng: JAX.random.PRNGKey
        """
        x, y = dataset

        indices = jax.random.permutation(shuffle_rng, y.shape[0])[:cutoff]
        obs, labels = x[indices], y[indices]

        obs = jnp.reshape(obs, data_shape)
        labels = jnp.reshape(labels, (-1, batch_size)) # should make batch size a global

        return (obs, labels)

    return jax.jit(_shuffle)




def rate_code(num_steps, max_r=0.75):
    """
    Unrolls input data along axis 1 and converts to rate encoded spikes; the probability of spiking is based on the input value multiplied by a max rate, with each time step being a sample drawn from a Bernoulli distribution.
    Currently Assumes input values have been rescaled to between 0 and 1.
    """

    if num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    if not 0.0 <= max_r <= 1.0:
        raise ValueError("max_r must be in [0, 1]")

    def _call(data, key):
        # Clip to a valid Bernoulli domain while preserving numerical precision.
        data = jnp.array(data, dtype=jnp.float32)
        data = jnp.clip(data, 0.0, 1.0)
        unrolled_data = jnp.repeat(data, num_steps, axis=1)
        probs = jnp.clip(unrolled_data * max_r, 0.0, 1.0)
        return jax.random.bernoulli(key, probs).astype(jnp.uint8)
    
    return jax.jit(_call)



def angle_code(neuron_count, min_val, max_val):
    """
    Higher-order-function which returns an angle encoding function; given a continuous value, an angle converter generates a one-hot vector corresponding to where the value falls between a specified minimum and maximum.
    To achieve non-linear descritization, apply a function to the continuous value before feeding it into the encoder.

    :neuron_count: The number of output channels for the angle encoder
    :min_val: A lower bound on the continuous input channel
    :max_val: An upper bound on the continuous input channel.
    """
    neurons = jnp.linspace(min_val, max_val, neuron_count)
        
    def _call(obs):
        digital = jnp.digitize(obs, neurons)
        return jax.nn.one_hot(digital, neuron_count)

    return jax.jit(_call)


def event_emulator(threshold=0.5):
    """
    Generates an event emulator that converts frame sequences into spikes using delta-threshold filtering.
    Useful for converting conventional video data into event-like spike representations.

    Each pixel accumulates temporal differences; when the absolute difference exceeds the threshold,
    a spike is emitted and the membrane is reset.

    :threshold: Delta threshold for spike emission. Typically 0.5-1.0 for normalized inputs.
    :return: Callable that takes frames (T, H, W, C) and returns spikes (T, H, W, C)
    """

    def _call(frames):
        """
        Convert frame sequence to spikes via delta-threshold.

        :frames: jnp.array of shape (T, H, W, C) with values in [0, 1]
        :return: jnp.array of shape (T, H, W, C) with binary spike output
        """
        frames = jnp.array(frames, dtype=jnp.float32)
        if frames.ndim != 4:
            raise ValueError(f"Expected 4D frames (T, H, W, C), got shape {frames.shape}")

        T, H, W, C = frames.shape
        spikes = jnp.zeros_like(frames, dtype=jnp.uint8)
        membrane = jnp.zeros((H, W, C), dtype=jnp.float32)

        def _step(carry, frame_t):
            membrane, spikes_seq = carry
            # Compute delta
            delta = jnp.abs(frame_t - membrane)
            # Emit spike if delta exceeds threshold
            spike_t = (delta > threshold).astype(jnp.uint8)
            # Reset membrane on spike
            membrane = membrane + spike_t * jnp.sign(frame_t - membrane) * threshold
            return (membrane, spike_t), spike_t

        _, spikes_all = jax.lax.scan(_step, (membrane, spikes[0]), frames)
        return spikes_all

    return jax.jit(_call)