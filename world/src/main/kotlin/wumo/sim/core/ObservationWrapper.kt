package wumo.sim.core

import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.t4

open class ObservationWrapper<O, OE : Any, A, AE : Any, WrappedEnv>(
    env: Env<O, OE, A, AE, WrappedEnv>) : Wrapper<O, OE, A, AE, WrappedEnv>(env) {
  
  override fun step(a: A): t4<O, Float, Boolean, Map<String, Any>> {
    val result = env.step(a)
    result._1 = observation(result._1)
    return result
  }
  
  override fun reset(): O {
    val obs = env.reset()
    return observation(obs)
  }
  
  open fun observation(frame: O): O = frame
}