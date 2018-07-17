package wumo.sim.util.ndarray

operator fun NDArray<Double>.unaryMinus(): NDArray<Double> {
  val c = copy()
  for (i in 0 until c.size)
    c[i] = -c[i]
  return c
}