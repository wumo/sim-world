package wumo.sim.util.ndarray.implementation

import wumo.sim.util.ndarray.Buf

class FloatArrayBuf(val raw: FloatArray) : Buf<Float> {
  override fun get(offset: Int): Float {
    return raw[offset]
  }
  
  override fun set(offset: Int, data: Float) {
    raw[offset] = data
  }
}