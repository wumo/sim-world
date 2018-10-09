package wumo.sim.util

import org.bytedeco.javacpp.*
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.types.*

inline val Pointer.isNotNull
  get() = !isNull

inline val <T : Any> NDArray<T>.native: BytePointer get() = raw.ptr
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

inline fun ShortPointer.toBytePointer(): BytePointer =
    BytePointer(this).capacity(capacity() * NDShort.byteSize)

inline fun BytePointer.toShortPointer(): ShortPointer =
    ShortPointer(this).capacity(capacity() / NDShort.byteSize)

inline fun IntPointer.toBytePointer(): BytePointer =
    BytePointer(this).capacity(capacity() * NDInt.byteSize)

inline fun BytePointer.toIntPointer(): IntPointer =
    IntPointer(this).capacity(capacity() / NDInt.byteSize)

inline fun LongPointer.toBytePointer(): BytePointer =
    BytePointer(this).capacity(capacity() * NDLong.byteSize)

inline fun BytePointer.toLongPointer(): LongPointer =
    LongPointer(this).capacity(capacity() / NDLong.byteSize)

inline fun FloatPointer.toBytePointer(): BytePointer =
    BytePointer(this).capacity(capacity() * NDFloat.byteSize)

inline fun BytePointer.toFloatPointer(): FloatPointer =
    FloatPointer(this).capacity(capacity() / NDFloat.byteSize)

inline fun DoublePointer.toBytePointer(): BytePointer =
    BytePointer(this).capacity(capacity() * NDDouble.byteSize)

inline fun BytePointer.toDoublePointer(): DoublePointer =
    DoublePointer(this).capacity(capacity() / NDDouble.byteSize)