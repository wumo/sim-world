package wumo.sim.util.ndarray.implementation

import wumo.sim.util.ndarray.Buf

class ArrayBuf<T>(val raw: Array<T>) : Buf<T> {
  override fun get(offset: Int): T = raw[offset]
  override fun set(offset: Int, data: T) {
    raw[offset] = data
  }
}