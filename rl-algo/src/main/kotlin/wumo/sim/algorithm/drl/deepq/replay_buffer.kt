package wumo.sim.algorithm.drl.deepq

import wumo.sim.util.Rand
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.t5
import wumo.sim.util.t7

open class ReplayBuffer<O, A>(val buffer_size: Int) {
  private val storage = ArrayList<t5<O, A, Float, O, Boolean>>(buffer_size)
  private var next_idx = 0
  
  val size: Int
    get() = storage.size
  
  fun add(obs: O, action: A, rew: Float, new_obs: O, done: Boolean) {
    val data = t5(obs, action, rew, new_obs, done)
    if (next_idx >= storage.size)
      storage += data
    else
      storage[next_idx] = data
    next_idx = (next_idx + 1) % buffer_size
  }
  
  /**
   * Sample a batch of experiences.
   */
  fun sample(batch_size: Int): t5<NDArray<*>, NDArray<*>, NDArray<Float>, NDArray<*>, NDArray<Float>> {
    val obses_t = ArrayList<O>(batch_size)
    val actions = ArrayList<A>(batch_size)
    val rewards = ArrayList<Float>(batch_size)
    val obses_tp1 = ArrayList<O>(batch_size)
    val dones = ArrayList<Float>(batch_size)
    repeat(batch_size) {
      val i = Rand().nextInt(0, storage.size)
      val (obs_t, action, reward, obs_tp1, done) = storage[i]
      obses_t += obs_t
      actions += action
      rewards += reward
      obses_tp1 += obs_tp1
      dones += if (done) 1f else 0f
    }
    return t5(NDArray.toNDArray(obses_t),
              NDArray.toNDArray(actions),
              NDArray.toNDArray(rewards) as NDArray<Float>,
              NDArray.toNDArray(obses_tp1),
              NDArray.toNDArray(dones) as NDArray<Float>)
  }
  
}

class PrioritizedReplayBuffer<O, A>(buffer_size: Int, alpha: Float) : ReplayBuffer<O, A>(buffer_size) {
  fun sample(batch_size: Int, beta: Float): t7<List<O>, List<A>, List<Double>, List<O>, List<Boolean>, NDArray<*>, Any> {
    TODO("not implemented")
  }
  
  fun update_priorities(batch_idxes: Any, new_priorities: NDArray<out Any>) {
    TODO("not implemented")
  }
}