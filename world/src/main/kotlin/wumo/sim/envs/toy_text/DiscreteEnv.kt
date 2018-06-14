package wumo.sim.envs.toy_text

import wumo.sim.core.Env
import wumo.sim.core.Space
import wumo.sim.spaces.Discrete
import wumo.sim.util.math.argmax
import wumo.sim.util.tuples.tuple4

typealias Transition = Array<Array<MutableList<tuple4<Double, Int, Double, Boolean>>>>

abstract class DiscreteEnv : Env<Int, Int> {
  var lastAction = -1
  abstract val nS: Int
  abstract val nA: Int
  abstract val P: Transition
  abstract val isd: DoubleArray
  override val action_space by lazy { Discrete(nA) }
  override val observation_space by lazy { Discrete(nS) }
  var s = -1
  
  override fun step(a: Int): tuple4<Int, Double, Boolean, Map<String, Any>> {
    val transitions = P[s][a]
    val i = argmax(0..transitions.lastIndex) { transitions[it]._1 }
    val (p, s, r, d) = transitions[i]
    this.s = s
    lastAction = a
    return tuple4(s, r, d, mapOf("prob" to p))
  }
  
  override fun reset(): Int {
    s = argmax(0 until nS) { isd[it] }
    lastAction = -1
    return s
  }
}