package wumo.sim.util.ndarray.implementation

import wumo.sim.util.ndarray.Buf

class BooleanArrayBuf(val raw: BooleanArray) : Buf<Boolean>() {
  override fun get(offset: Int) = raw[offset]
  
  override fun set(offset: Int, data: Boolean) {
    raw[offset] = data
  }
  
  override fun copy() = BooleanArrayBuf(raw.clone())
  
  override fun slice(start: Int, end: Int): Buf<Boolean> =
      BooleanArrayBuf(raw.sliceArray(start until end))
  
  override val size = raw.size
}
