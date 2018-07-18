package wumo.sim.algorithm.drl.deepq

import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.util.Rand
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.tuple5
import wumo.sim.util.tuple7

open class ReplayBuffer<O, A>(val buffer_size: Int) {
  private val storage = ArrayList<tuple5<O, A, Double, O, Boolean>>(buffer_size)
  private var next_idx = 0
  
  fun add(obs: O, action: A, rew: Double, new_obs: O, done: Boolean) {
    val data = tuple5(obs, action, rew, new_obs, done)
    if (next_idx >= storage.size)
      storage += data
    else
      storage[next_idx] = data
    next_idx = (next_idx + 1) % buffer_size
  }
  
  fun sample(batch_size: Int, beta: Float): tuple7<Any, Any, Any, Any, Any, Any, Any> {
    TODO("not implemented")
  }
  
  /**
   * Sample a batch of experiences.
   */
  fun sample(batch_size: Int): tuple5<List<O>, List<A>, List<Double>, List<O>, List<Boolean>> {
    val obses_t = ArrayList<O>(batch_size)
    val actions = ArrayList<A>(batch_size)
    val rewards = ArrayList<Double>(batch_size)
    val obses_tp1 = ArrayList<O>(batch_size)
    val dones = ArrayList<Boolean>(batch_size)
    repeat(batch_size) {
      val i = Rand().nextInt(0, storage.size)
      val (obs_t, action, reward, obs_tp1, done) = storage[i]
      obses_t += obs_t
      actions += action
      rewards += reward
      obses_tp1 += obs_tp1
      dones += done
    }
    return tuple5(obses_t, actions, rewards, obses_tp1, dones)
  }
  
  fun update_priorities(batch_idxes: Any, new_priorities: NDArray<out Any?>) {
    TODO("not implemented")
  }
}