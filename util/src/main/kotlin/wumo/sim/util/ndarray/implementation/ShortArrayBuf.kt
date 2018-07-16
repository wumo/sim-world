package wumo.sim.util.ndarray.implementation

import wumo.sim.util.ndarray.Buf

class ShortArrayBuf(val raw: ShortArray) : Buf<Short> {
  override fun get(offset: Int): Short {
    return raw[offset]
  }
  
  override fun set(offset: Int, data: Short) {
    raw[offset] = data
  }
}