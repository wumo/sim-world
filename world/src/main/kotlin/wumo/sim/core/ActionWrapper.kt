package wumo.sim.core

import wumo.sim.util.t4

open class ActionWrapper<O, OE : Any, A, AE : Any, WrappedEnv>(
    env: Env<O, OE, A, AE, WrappedEnv>) : Wrapper<O, OE, A, AE, WrappedEnv>(env) {
  
  override fun step(a: A): t4<O, Float, Boolean, Map<String, Any>> {
    val a = action(a)
    return env.step(a)
  }
  
  open fun action(action: A): A = action
  
  open fun reverse_action(action: A): A = action
}