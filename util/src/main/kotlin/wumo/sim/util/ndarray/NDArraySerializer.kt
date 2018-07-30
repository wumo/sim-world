package wumo.sim.util.ndarray

import okio.BufferedSink
import okio.BufferedSource
import wumo.sim.util.Dimension
import wumo.sim.util.SwitchType3
import wumo.sim.util.SwitchValue2
import wumo.sim.util.ndarray.implementation.*

fun BufferedSink.encode(v: Float) = writeInt(v.toBits())
fun BufferedSource.decodeFloat() = Float.fromBits(readInt())
fun BufferedSink.encode(v: Double) = writeLong(v.toBits())
fun BufferedSource.decodeDouble() = Double.fromBits(readLong())
fun BufferedSink.encode(v: Boolean) = writeByte(if (v) 1 else 0)
fun BufferedSource.decodeBoolean() = readByte() != 0.toByte()
fun BufferedSink.encode(v: Byte) = writeByte(v.toInt())
fun BufferedSource.decodeByte() = readByte()
fun BufferedSink.encode(v: Short) = writeShort(v.toInt())
fun BufferedSource.decodeShort() = readShort()
fun BufferedSink.encode(v: Int) = writeInt(v)
fun BufferedSource.decodeInt() = readInt()
fun BufferedSink.encode(v: Long) = writeLong(v)
fun BufferedSource.decodeLong() = readLong()

fun BufferedSink.encode(b: IntArray) {
  writeInt(b.size)
  for (i in b)
    writeInt(i)
}

fun BufferedSource.decodeIntArray(): IntArray {
  val size = readInt()
  return IntArray(size) {
    readInt()
  }
}

fun BufferedSink.encode(b: ByteArray) {
  writeInt(b.size)
  write(b)
}

fun BufferedSource.decodeByteArray(): ByteArray {
  val size = readInt()
  return readByteArray(size.toLong())
}

fun BufferedSink.encode(s: String) {
  val bytes = s.toByteArray()
  encode(bytes)
}

fun BufferedSource.decodeString() = String(decodeByteArray())

private val saveSwtich = SwitchType3<BufferedSink, NDArray<*>, Unit>().apply {
  case<Float> { _2.writeInt(0); _3.forEach { _2.encode(it as Float) } }
  case<Double> { _2.writeInt(1);_3.forEach { _2.encode(it as Double) } }
  case<Boolean> { _2.writeInt(2);_3.forEach { _2.encode(it as Boolean) } }
  case<Byte> { _2.writeInt(3);_3.forEach { _2.encode(it as Byte) } }
  case<Short> { _2.writeInt(4);_3.forEach { _2.encode(it as Short) } }
  case<Int> { _2.writeInt(5);_3.forEach { _2.encode(it as Int) } }
  case<Long> { _2.writeInt(6);_3.forEach { _2.encode(it as Long) } }
  case<String> { _2.writeInt(7); _3.forEach { _2.encode(it as String) } }
}
private val loadSwtich = SwitchValue2<Int, BufferedSource, Int, Buf<*>>().apply {
  case(0) { FloatArrayBuf(FloatArray(_2) { _1.decodeFloat() }) }
  case(1) { DoubleArrayBuf(DoubleArray(_2) { _1.decodeDouble() }) }
  case(2) { BooleanArrayBuf(BooleanArray(_2) { _1.decodeBoolean() }) }
  case(3) { ByteArrayBuf(ByteArray(_2) { _1.decodeByte() }) }
  case(4) { ShortArrayBuf(ShortArray(_2) { _1.decodeShort() }) }
  case(5) { IntArrayBuf(IntArray(_2) { _1.decodeInt() }) }
  case(6) { LongArrayBuf(LongArray(_2) { _1.decodeLong() }) }
  case(7) { ArrayBuf(Array(_2) { _1.decodeString() }) }
}

fun BufferedSource.decodeNDArray(): NDArray<*> {
  val shape = Dimension(decodeIntArray())
  val size = shape.numElements()
  val type = readInt()
  val buf = loadSwtich(type, this, size)
  return NDArray(shape, buf as Buf<Any>)
}

fun BufferedSink.encode(obj: NDArray<*>) {
  encode(obj.shape.asIntArray())
  val first = obj.first()
  saveSwtich(first, this, obj)
}