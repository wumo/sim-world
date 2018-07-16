package wumo.sim.util.ndarray.implementation

import wumo.sim.util.ndarray.Buf

class ByteArrayBuf(val raw: ByteArray) : Buf<Byte> {
  override fun get(offset: Int): Byte {
    return raw[offset]
  }
  
  override fun set(offset: Int, data: Byte) {
    raw[offset] = data
  }
}