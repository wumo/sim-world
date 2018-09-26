package wumo.sim.algorithm.drl.common

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.Pointer.memcpy
import wumo.sim.util.ndarray.Buf

class BytePointerBuf(val raw: BytePointer) : Buf<Byte>() {
  override fun get(offset: Int): Byte =
      raw[offset.toLong()]
  
  override fun set(offset: Int, data: Byte) {
    raw.put(offset.toLong(), data)
  }
  
  override fun copy(): Buf<Byte> {
    val dst = BytePointer(raw.limit())
    memcpy(dst, raw, raw.limit())
    return BytePointerBuf(dst)
  }
  
  override fun slice(start: Int, end: Int): Buf<Byte> {
    TODO("not implemented")
  }
  
  override val size: Int = raw.limit().toInt()
}