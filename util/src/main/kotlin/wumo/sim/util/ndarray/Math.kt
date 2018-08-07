package wumo.sim.util.ndarray

import wumo.sim.util.Shape
import wumo.sim.util.SwitchType2
import wumo.sim.util.ndarray.NDArray.Companion.toNDArray

operator fun NDArray<Float>.unaryMinus(): NDArray<Float> {
  val c = copy()
  for (i in 0 until c.size)
    c[i] = -c[i]
  return c
}

operator fun <T : Any> NDArray<T>.plus(b: NDArray<T>): NDArray<T> {
  TODO()
}

operator fun <T : Any> NDArray<T>.plus(b: Number): NDArray<T> {
  TODO()
}

fun <T : Any> abs(a: NDArray<T>): NDArray<T> {
  TODO()
}

val ones_like_switch = SwitchType2<Shape, NDArray<*>>().apply {
  case<Float> { NDArray(_2, 1f) }
  case<Double> { NDArray(_2, 1.0) }
  case<Byte> { NDArray(_2, 1.toByte()) }
  case<Short> { NDArray(_2, 1.toShort()) }
  case<Int> { NDArray(_2, 1) }
  case<Long> { NDArray(_2, 1L) }
}

fun <T : Number> ones_like(a: NDArray<T>) = ones_like_switch(a.first(), a.shape) as NDArray<T>

fun <T : Any> newaxis(a: NDArray<T>) = toNDArray(arrayOf(a))