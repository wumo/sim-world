package wumo.sim.envs.toy_text

import wumo.sim.core.Env
import wumo.sim.spaces.Discrete
import wumo.sim.util.argmax
import wumo.sim.util.t4
import wumo.sim.utils.np_random
import kotlin.random.Random

typealias Transition = Array<Array<MutableList<t4<Float, Int, Float, Boolean>>>>

abstract class DiscreteEnv(val nS: Int,
                           val nA: Int,
                           val P: Transition,
                           val isd: FloatArray) : Env<Int, Int> {
  
  var lastAction = -1
  var s = -1
  override val action_space = Discrete(nA)
  override val observation_space = Discrete(nS)
  
  lateinit var rand: Random
  
  init {
    seed()
    reset()
  }
  
  override fun step(a: Int): t4<Int, Float, Boolean, Map<String, Any>> {
    val transitions = P[s][a]
    val i = argmax(0..transitions.lastIndex) { transitions[it]._1 }
    val (p, s, r, d) = transitions[i]
    this.s = s
    lastAction = a
    return t4(s, r, d, mapOf("prob" to p))
  }
  
  override fun reset(): Int {
    s = argmax(0 until nS) { isd[it] }
    lastAction = -1
    return s
  }
  
  override fun seed(seed: Long?): List<Long> {
    val (rand, seed) = np_random(seed)
    this.rand = rand
    return listOf(seed)
  }
}