package wumo.sim.util

import org.bytedeco.javacpp.BytePointer
import org.junit.Test
import wumo.buf.buf

class BufTest {
  @Test
  fun testBuf() {
    buf.buf_init()
    val src=BytePointer(4L)
    src.putInt(123)
    val a=src.getInt(0L)
    println(a)
  }
}