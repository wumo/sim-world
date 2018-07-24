package wumo.sim.algorithm.tensorflow.util

import org.bytedeco.javacpp.IntPointer

fun IntPointer.toArray(): IntArray {
  val size = limit()
  return IntArray(size.toInt()) {
    get(it.toLong())
  }
}