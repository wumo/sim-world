package wumo.sim.algorithm.drl.deepq

import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.t7

class PrioritizedReplayBuffer<O, A>(buffer_size: Int, alpha: Float) : ReplayBuffer<O, A>(buffer_size) {
  fun sample(batch_size: Int, beta: Float): t7<List<O>, List<A>, List<Double>, List<O>, List<Boolean>, NDArray<*>, Any> {
    TODO("not implemented")
  }
  
  fun update_priorities(batch_idxes: Any, new_priorities: NDArray<out Any>) {
    TODO("not implemented")
  }
}