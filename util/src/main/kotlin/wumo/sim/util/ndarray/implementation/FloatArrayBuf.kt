package wumo.sim.util.ndarray.implementation

import wumo.sim.util.ndarray.Buf

class FloatArrayBuf(val raw: FloatArray) : Buf<Float>() {
  override fun get(offset: Int): Float {
    return raw[offset]
  }
  
  override fun set(offset: Int, data: Float) {
    raw[offset] = data
  }
  
  override fun copy() = FloatArrayBuf(raw.clone())
  
  override fun slice(start: Int, end: Int): Buf<Float> =
      FloatArrayBuf(raw.sliceArray(start until end))
  
  override val size = raw.size
}