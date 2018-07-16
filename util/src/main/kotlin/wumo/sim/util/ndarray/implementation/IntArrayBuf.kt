package wumo.sim.util.ndarray.implementation

import wumo.sim.util.ndarray.Buf

class IntArrayBuf(val raw: IntArray) : Buf<Int> {
  override fun get(offset: Int): Int {
    return raw[offset]
  }
  
  override fun set(offset: Int, data: Int) {
    raw[offset] = data
  }
}