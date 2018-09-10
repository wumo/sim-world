package wumo.sim.util.ndarray.implementation

import wumo.sim.util.ndarray.Buf

class DoubleArrayBuf(val raw: DoubleArray) : Buf<Double> {
  override fun get(offset: Int): Double {
    return raw[offset]
  }
  
  override fun set(offset: Int, data: Double) {
    raw[offset] = data
  }
  
  override fun copy() = DoubleArrayBuf(raw.clone())
  
  override fun slice(start: Int, end: Int): Buf<Double> =
      DoubleArrayBuf(raw.sliceArray(start until end))
  
  override val size = raw.size
}