package wumo.sim.util.ndarray

import org.bytedeco.javacpp.BytePointer
import org.bytedeco.javacpp.Pointer.memcpy
import wumo.sim.util.ndarray.types.NDType

@Suppress("NOTHING_TO_INLINE")
open class BytePointerBuf<T : Any>(val ptr: BytePointer,
                                   val ndType: NDType<T>) : Iterable<T> {
  
  companion object {
    inline operator fun <T : Any> invoke(
        size: Int,
        dtype: NDType<T>,
        init: (Int) -> T = { dtype.zero() }): BytePointerBuf<T> {
      val byteSize = dtype.byteSize
      val raw = BytePointer((size * byteSize).toLong())
      for (i in 0 until size)
        dtype.put(raw, (i * byteSize).toLong(), init(i))
      return BytePointerBuf(raw, dtype)
    }
  }
  
  private val byteSize = ndType.byteSize
  private inline fun offset(idx: Int): Long = (idx * byteSize).toLong()
  
  open operator fun get(idx: Int): T = ndType.get(ptr, offset(idx))
  
  open operator fun set(idx: Int, data: T) = ndType.put(ptr, offset(idx), data)
  
  open fun copy(): BytePointerBuf<T> {
    val dst = BytePointer(ptr.capacity())
    memcpy(dst, ptr, ptr.capacity())
    return BytePointerBuf(dst, ndType)
  }
  
  open fun slice(start: Int, end: Int): BytePointerBuf<T> {
    val start = offset(start)
    val end = offset(end)
    
    val dst = BytePointer(end - start)
    memcpy(dst, ptr.position(start), dst.capacity())
    return BytePointerBuf(dst, ndType)
  }
  
  open val size: Int = (ptr.capacity() / byteSize).toInt()
  
  override fun iterator() = object : Iterator<T> {
    var a = 0
    override fun hasNext() = a < size
    
    override fun next() = get(a++)
  }
  
  override fun toString(): String =
      this.joinToString(",")
}