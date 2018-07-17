package wumo.sim.util.ndarray.implementation

import wumo.sim.util.ndarray.Buf

class LongArrayBuf(val raw: LongArray) : Buf<Long> {
  override fun get(offset: Int): Long {
    return raw[offset]
  }
  
  override fun set(offset: Int, data: Long) {
    raw[offset] = data
  }
  
  override fun copy() = LongArrayBuf(raw.clone())
  
  override val size = raw.size
}