package wumo.sim.util.ndarray

operator fun NDArray<Double>.unaryMinus(): NDArray<Double> {
  val c = copy()
  for (i in 0 until c.size)
    c[i] = -c[i]
  return c
}

operator fun <T> NDArray<T>.plus(b: NDArray<T>): NDArray<T> {
  TODO()
}

operator fun <T> NDArray<T>.plus(b: Number): NDArray<T> {
  TODO()
}

fun <T> abs(a: NDArray<T>): NDArray<T> {
  TODO()
}

fun <T> ones_like(a: NDArray<T>): NDArray<T> {
  TODO()
}