package wumo.sim.core

import wumo.sim.util.t4

open class RewardWrapper<O, OE : Any, A, AE : Any, WrappedEnv>(
    env: Env<O, OE, A, AE, WrappedEnv>) : Wrapper<O, OE, A, AE, WrappedEnv>(env) {
  
  override fun step(a: A): t4<O, Float, Boolean, Map<String, Any>> {
    val result = env.step(a)
    result._2 = reward(result._2)
    return result
  }
  
  open fun reward(frame: Float): Float = frame
}