package wumo.sim.util.ndarray

import wumo.sim.util.*
import wumo.sim.util.ndarray.NDArray.Companion.toNDArray

operator fun NDArray<Float>.unaryMinus(): NDArray<Float> {
  val c = copy()
  for (i in 0 until c.size)
    c[i] = -c[i]
  return c
}

operator fun NDArray<Float>.timesAssign(scale: Float) {
  flatten().forEach { (i, v) ->
    rawSet(i, v * scale)
  }
}

operator fun <T : Any> NDArray<T>.plus(b: NDArray<T>): NDArray<T> {
  TODO()
}

operator fun <T : Any> NDArray<T>.plus(b: Number): NDArray<T> {
  TODO()
}

operator fun NDArray<Float>.divAssign(b: Float) {
  for (i in 0 until raw.size)
    raw[i] /= b
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

fun randomNormal(mean: Float = 0f, scale: Float = 1f, shape: Shape): NDArray<Float> =
    NDArray(shape, FloatArray(shape.numElements()) {
      Rand().nextGaussian(mean, scale)
    })

fun <T : Any> randomChoice(a: NDArray<T>, p: NDArray<out Number>): T {
  val _p = DoubleArray(p.size) { p[it].toDouble() }
  val base = _p.sum()
  val chosen = Rand().nextDouble(0.0, base)
  var acc = 0.0
  for ((i, v) in _p.withIndex()) {
    acc += v
    if (acc > chosen) return a[i]
  }
  NONE()
}

fun randomChoice(p: NDArray<out Number>): Int {
  val _p = DoubleArray(p.size) { p[it].toDouble() }
  val base = _p.sum()
  val chosen = Rand().nextDouble(0.0, base)
  var acc = 0.0
  for ((i, v) in _p.withIndex()) {
    acc += v
    if (acc > chosen) return i
  }
  NONE()
}

fun arrayEqual(a: NDArray<*>, b: NDArray<*>): Boolean {
  if (a.shape != b.shape) return false
  val na = a.numElements
  for (i in 0 until na)
    if (a.rawGet(i) != b.rawGet(i))
      return false
  return true
}

fun <T : Any> NDArray<T>.reduce(axis: Int = 0,
                                accumulator: (T, T) -> T): NDArray<T> {
  val origin = this
  val shape = shape
  if (isScalar) return copy()
  require(shape.rank >= 1)
  val resultShape = Shape(List(shape.rank - 1) {
    if (it < axis) shape[it]
    else shape[it + 1]
  })
  val N = shape[axis]
  val idx = IntArray(shape.rank)
  return NDArray.from<Any>(resultShape) {
    if (axis > 0) System.arraycopy(it, 0, idx, 0, axis - 1)
    System.arraycopy(it, axis, idx, axis + 1, it.size - axis)
    idx[axis] = 0
    var acc = origin.get(*idx)
    for (i in 1 until N) {
      idx[axis] = i
      val element = origin.get(*idx)
      acc = accumulator(acc, element)
    }
    acc
  } as NDArray<T>
}

fun <T> NDArray<T>.max(axis: Int = 0): NDArray<T>
    where T : Number, T : Comparable<T> =
    reduce(axis) { a, b -> if (a > b) a else b }

fun <T> NDArray<T>.min(axis: Int = 0): NDArray<T>
    where T : Number, T : Comparable<T> =
    reduce(axis) { a, b -> if (a < b) a else b }

inline fun <reified R : Number, reified T : Number> NDArray<T>.cast(): NDArray<R> {
  TODO()
}