package wumo.sim.world.examples.algorithm

import wumo.sim.util.ndarray.NDArray

class LinearTileCodingFunc(val feature: SuttonTileCoding) {
  val w = FloatArray(feature.numOfComponents)
  operator fun invoke(s: NDArray<Float>, a: Int): Float {
    return w.innerProduct(feature(s, a))
  }
}