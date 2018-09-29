package wumo.sim.tensorflow.util

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.PointerScope

inline val Pointer.isNotNull
  get() = !isNull

inline fun <R> native(block: () -> R): R {
  val ptrScope = PointerScope()
  try {
    return block()
  } finally {
    ptrScope.close()
  }
}