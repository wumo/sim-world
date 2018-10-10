package wumo.sim.algorithm.drl.deepq

import wumo.sim.core.Space
import wumo.sim.util.*
import wumo.sim.util.ndarray.BytePointerBuf
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.types.NDFloat
import wumo.sim.util.ndarray.types.NDType

open class ReplayBuffer<O : Any, OE : Any, A : Any, AE : Any>(val buffer_size: Int,
                                                              val observation_space: Space<O, OE>,
                                                              val action_space: Space<A, AE>) {
  
  private val storage = ArrayList<t5<O, A, Float, O, Boolean>>(buffer_size)
  private var next_idx = 0
  
  val size: Int
    get() = storage.size
  
  fun add(obs: O, action: A, rew: Float, new_obs: O, done: Boolean) {
    val data = t5(obs, action, rew, new_obs, done)
    (new_obs as? NDArray<*>)?.ref()
    if (next_idx >= storage.size)
      storage += data
    else {
      val pre = storage[next_idx]._1
      if (pre is NDArray<*>)
        pre.unref()
      storage[next_idx] = data
    }
    next_idx = (next_idx + 1) % buffer_size
  }
  
  val obs_t_ThreadLocal = ThreadLocal<NDArray<OE>>()
  val obs_tp1_ThreadLocal = ThreadLocal<NDArray<OE>>()
  val action_ThreadLocal = ThreadLocal<NDArray<AE>>()
  val reward_ThreadLocal = ThreadLocal<NDArray<Float>>()
  val dones_ThreadLocal = ThreadLocal<NDArray<Float>>()
  
  fun <E : Any> make(batch_size: Int, shape: Shape, dtype: NDType<E>): NDArray<E> {
    val shape = batch_size + shape
    return NDArray(shape, BytePointerBuf(shape.numElements(), dtype))
  }
  
  /**
   * Sample a batch of experiences.
   */
  fun sample(batch_size: Int): t5<NDArray<OE>, NDArray<AE>, NDArray<Float>, NDArray<OE>, NDArray<Float>> {
    val obses_t = obs_t_ThreadLocal.get() ?: make(batch_size, observation_space.shape, observation_space.dtype)
    val actions = action_ThreadLocal.get() ?: make(batch_size, action_space.shape, action_space.dtype)
    val rewards = reward_ThreadLocal.get() ?: make(batch_size, scalarDimension, NDFloat)
    val obses_tp1 = obs_tp1_ThreadLocal.get() ?: make(batch_size, observation_space.shape, observation_space.dtype)
    val dones = dones_ThreadLocal.get() ?: make(batch_size, scalarDimension, NDFloat)
    native {
      repeat(batch_size) {
        val i = Rand().nextInt(0, storage.size)
        val (obs_t, action, reward, obs_tp1, done) = storage[i]
        obses_t[it] = NDArray.toNDArray(obs_t)
        actions[it] = NDArray.toNDArray(action)
        rewards[it] = NDArray.toNDArray(reward)
        obses_tp1[it] = NDArray.toNDArray(obs_tp1)
        dones[it] = NDArray.toNDArray(if (done) 1f else 0f)
      }
    }
    return t5(obses_t, actions, rewards, obses_tp1, dones)
  }
}

class PrioritizedReplayBuffer<O : Any, OE : Any, A : Any, AE : Any>
constructor(buffer_size: Int,
            observation_space: Space<O, OE>,
            action_space: Space<A, AE>,
            alpha: Float)
  : ReplayBuffer<O, OE, A, AE>(buffer_size, observation_space, action_space) {
  
  fun sample(batch_size: Int, beta: Float)
      : t7<List<O>, List<A>, List<Double>, List<O>, List<Boolean>, NDArray<*>, Any> {
    TODO("not implemented")
  }
  
  fun update_priorities(batch_idxes: Any, new_priorities: NDArray<out Any>) {
    TODO("not implemented")
  }
}