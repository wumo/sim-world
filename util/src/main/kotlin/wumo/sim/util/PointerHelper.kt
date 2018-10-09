package wumo.sim.util

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.PointerScope
import wumo.sim.util.ndarray.NDArray

inline val Pointer.isNotNull
  get() = !isNull

inline val <T : Any> NDArray<T>.native: BytePointer get() = buf.ptr
inline fun <T : Any> NDArray<T>.ref(): NDArray<T> {
  PointerScope.getInnerScope().detach(native)
  return this
}

inline fun <T : Any> NDArray<T>.unref(): NDArray<T> {
  PointerScope.getInnerScope().attach(native)
  return this
}

inline fun Pointer.unref(): Pointer {
  PointerScope.getInnerScope().attach(this)
  return this
}

inline fun Pointer.ref(): Pointer {
  PointerScope.getInnerScope().detach(this)
  return this
}

inline fun <R> native(block: () -> R): R {
  val ptrScope = PointerScope()
  try {
    return block()
  } finally {
    ptrScope.close()
  }
}