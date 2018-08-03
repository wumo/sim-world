package wumo.sim.algorithm.tensorflow.util

import org.bytedeco.javacpp.IntPointer
import org.bytedeco.javacpp.Pointer

inline val Pointer.isNotNull
  get() = !isNull

fun IntPointer.toArray(): IntArray {
  val size = limit()
  return IntArray(size.toInt()) {
    get(it.toLong())
  }
}