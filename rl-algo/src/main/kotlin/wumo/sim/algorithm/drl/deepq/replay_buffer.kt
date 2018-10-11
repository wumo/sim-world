package wumo.sim.algorithm.drl.deepq

import org.apache.commons.math3.util.FastMath
import wumo.sim.algorithm.drl.common.MinSegmentTree
import wumo.sim.algorithm.drl.common.SumSegmentTree
import wumo.sim.core.Space
import wumo.sim.util.*
import wumo.sim.util.ndarray.BytePointerBuf
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.toNDArray
import wumo.sim.util.ndarray.types.NDFloat
import wumo.sim.util.ndarray.types.NDType

open class ReplayBuffer<O : Any, OE : Any, A : Any, AE : Any>(val buffer_size: Int,
                                                              val observation_space: Space<O, OE>,
                                                              val action_space: Space<A, AE>) {
  
  protected val storage = ArrayList<t5<O, A, Float, O, Boolean>>(buffer_size)
  protected var next_idx = 0
  
  val size: Int
    get() = storage.size
  
  open fun add(obs: O, action: A, rew: Float, new_obs: O, done: Boolean) {
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
  fun sample(batch_size: Int) =
      encode_sample(List(batch_size) { Rand().nextInt(0, storage.size) })
  
  fun encode_sample(idxes: List<Int>): t5<NDArray<OE>, NDArray<AE>, NDArray<Float>, NDArray<OE>, NDArray<Float>> {
    val batch_size = idxes.size
    val obses_t = obs_t_ThreadLocal.get() ?: make(batch_size, observation_space.shape, observation_space.dtype)
    val actions = action_ThreadLocal.get() ?: make(batch_size, action_space.shape, action_space.dtype)
    val rewards = reward_ThreadLocal.get() ?: make(batch_size, scalarDimension, NDFloat)
    val obses_tp1 = obs_tp1_ThreadLocal.get() ?: make(batch_size, observation_space.shape, observation_space.dtype)
    val dones = dones_ThreadLocal.get() ?: make(batch_size, scalarDimension, NDFloat)
    native {
      for (i in 0 until idxes.size) {
        val (obs_t, action, reward, obs_tp1, done) = storage[idxes[i]]
        obses_t[i] = NDArray.toNDArray(obs_t)
        actions[i] = NDArray.toNDArray(action)
        rewards[i] = NDArray.toNDArray(reward)
        obses_tp1[i] = NDArray.toNDArray(obs_tp1)
        dones[i] = NDArray.toNDArray(if (done) 1f else 0f)
      }
    }
    return t5(obses_t, actions, rewards, obses_tp1, dones)
  }
}

class PrioritizedReplayBuffer<O : Any, OE : Any, A : Any, AE : Any>
constructor(buffer_size: Int,
            observation_space: Space<O, OE>,
            action_space: Space<A, AE>,
            val alpha: Float)
  : ReplayBuffer<O, OE, A, AE>(buffer_size, observation_space, action_space) {
  
  private var it_sum: SumSegmentTree
  private var it_min: MinSegmentTree
  
  init {
    var it_capacity = 1
    while (it_capacity < buffer_size)
      it_capacity *= 2
    it_sum = SumSegmentTree(it_capacity)
    it_min = MinSegmentTree(it_capacity)
  }
  
  var max_priority = 1f
  
  override fun add(obs: O, action: A, rew: Float, new_obs: O, done: Boolean) {
    val idx = next_idx
    super.add(obs, action, rew, new_obs, done)
    it_sum[idx] = FastMath.pow(max_priority.toDouble(), alpha.toDouble()).toFloat()
    it_min[idx] = FastMath.pow(max_priority.toDouble(), alpha.toDouble()).toFloat()
  }
  
  fun sampel_proportional(batch_size: Int): List<Int> {
    val p_total = it_sum.sum(0, storage.size - 1)
    val every_range_len = p_total / batch_size
    return List(batch_size) { i ->
      val mass = Rand().nextFloat() * every_range_len + i * every_range_len
      val idx = it_sum.find_prefixsum_idx(mass)
      idx
    }
  }
  
  fun sample(batch_size: Int, beta: Float)
      : t7<NDArray<OE>, NDArray<AE>, NDArray<Float>,
      NDArray<OE>, NDArray<Float>, NDArray<Float>, List<Int>> {
    assert(beta > 0)
    val idxes = sampel_proportional(batch_size)
    val p_min = it_min.min() / it_sum.sum()
    val max_weight = FastMath.pow(p_min * storage.size.toDouble(), -beta.toDouble())
    val weights = idxes.map { idx ->
      val p_sample = it_sum[idx] / it_sum.sum()
      (FastMath.pow(p_sample * storage.size.toDouble(),
                    -beta.toDouble()) / max_weight).toFloat()
    }.toNDArray<Float>()
    val (obses_t, actions, rewards, obses_tp1, dones) = encode_sample(idxes)
    return t7(obses_t, actions, rewards, obses_tp1, dones, weights, idxes)
  }
  
  fun update_priorities(batch_idxes: List<Int>, new_priorities: NDArray<Float>) {
    assert(batch_idxes.size == new_priorities.size)
    for ((idx, priority) in batch_idxes.zip(new_priorities)) {
      assert(priority > 0)
      assert(idx in 0 until storage.size)
      it_sum[idx] = FastMath.pow(priority.toDouble(), alpha.toDouble()).toFloat()
      it_min[idx] = FastMath.pow(priority.toDouble(), alpha.toDouble()).toFloat()
      max_priority = maxOf(max_priority, priority)
    }
  }
}