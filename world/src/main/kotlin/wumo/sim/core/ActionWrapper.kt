package wumo.sim.core

import wumo.sim.util.t4

open class ActionWrapper<O, A, WrappedEnv>(
    env: Env<O, A, WrappedEnv>) : Wrapper<O, A, WrappedEnv>(env) {
  
  override fun step(a: A): t4<O, Float, Boolean, Map<String, Any>> {
    val a = action(a)
    return env.step(a)
  }
  
  open fun action(action: A): A = action
  
  open fun reverse_action(action: A): A = action
}