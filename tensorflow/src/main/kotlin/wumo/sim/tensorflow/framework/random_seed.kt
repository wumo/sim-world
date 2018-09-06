package wumo.sim.tensorflow.framework

import wumo.sim.tensorflow.tf
import wumo.sim.util.t2

val DEFAULT_GRAPH_SEED = 87654321

fun getSeed(opSeed: Int? = null): t2<out Int?, out Int?> {
  val globalSeed = tf.currentGraph.randomSeed
  var opSeed = opSeed
  val seeds = if (globalSeed != null) {
    if (opSeed == null)
      opSeed = tf.currentGraph.nextIdCounter
    t2(globalSeed, opSeed)
  } else {
    if (opSeed != null)
      t2(DEFAULT_GRAPH_SEED, opSeed)
    else
      t2(null, null)
  }
  return if (seeds._1 == 0 && seeds._2 == 0)
    t2(0, Int.MAX_VALUE.toInt())
  else seeds
}