package wumo.sim.world.examples.algorithm

import wumo.sim.util.ndarray.NDArray

class LinearTileCodingFunc(val feature: SuttonTileCoding) {
  val w = DoubleArray(feature.numOfComponents)
  operator fun invoke(s: NDArray<Double>, a: Int): Double {
    return w.innerProduct(feature(s, a))
  }
}