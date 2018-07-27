package wumo.sim.util.ndarray

import wumo.sim.util.Dimension
import wumo.sim.util.SwitchType3
import wumo.sim.util.SwitchValue2
import wumo.sim.util.ndarray.implementation.*
import java.io.InputStream
import java.io.OutputStream

object NDArraySerializer {
  class writer {
    fun write(value: Int) {
      
    }
    
    fun write(value: Float) {
      TODO("not implemented")
    }
    
    fun write(value: Double) {
      TODO("not implemented")
    }
    
    fun write(value: Boolean) {
      
    }
    
    fun write(value: Byte) {
      
    }
    
    fun write(sh: Short) {
      TODO("not implemented")
    }
    
    fun write(l: Long) {
      TODO("not implemented")
    }
    
    fun write(s: String) {
      
      
    }
    
    fun write(elements: List<Int>) {
      TODO("not implemented")
    }
  }
  
  class reader {
    fun readFloatValue(): Float {
      TODO("not implemented")
    }
    
    fun readDoubleValue(): Double {
      TODO("not implemented")
    }
    
    fun readBooleanValue(): Boolean {
      TODO("not implemented")
    }
    
    fun readByteValue(): Byte {
      TODO("not implemented")
    }
    
    fun readShortValue(): Short {
      TODO("not implemented")
    }
    
    fun readIntValue(): Int {
      TODO("not implemented")
    }
    
    fun readLongValue(): Long {
      TODO("not implemented")
    }
    
    fun readStringValue(): Any {
      TODO("not implemented")
    }
    
    fun readList(): MutableList<Int> {
      TODO("not implemented")
    }
    
  }
  
  private val saveSwtich = SwitchType3<writer, NDArray<*>, Unit>().apply {
    case<Float> { _2.write(0); _3.forEach { _2.write(it as Float) } }
    case<Double> { _2.write(1);_3.forEach { _2.write(it as Double) } }
    case<Boolean> { _2.write(2);_3.forEach { _2.write(it as Boolean) } }
    case<Byte> { _2.write(3);_3.forEach { _2.write(it as Byte) } }
    case<Short> { _2.write(4);_3.forEach { _2.write(it as Short) } }
    case<Int> { _2.write(5);_3.forEach { _2.write(it as Int) } }
    case<Long> { _2.write(6);_3.forEach { _2.write(it as Long) } }
    case<String> { _2.write(7); _3.forEach { _2.write(it as String) } }
  }
  private val loadSwtich = SwitchValue2<Int, reader, Int, Buf<*>>().apply {
    case(0) { FloatArrayBuf(FloatArray(_2) { _1.readFloatValue() }) }
    case(1) { DoubleArrayBuf(DoubleArray(_2) { _1.readDoubleValue() }) }
    case(2) { BooleanArrayBuf(BooleanArray(_2) { _1.readBooleanValue() }) }
    case(3) { ByteArrayBuf(ByteArray(_2) { _1.readByteValue() }) }
    case(4) { ShortArrayBuf(ShortArray(_2) { _1.readShortValue() }) }
    case(5) { IntArrayBuf(IntArray(_2) { _1.readIntValue() }) }
    case(6) { LongArrayBuf(LongArray(_2) { _1.readLongValue() }) }
    case(7) { ArrayBuf(Array(_2) { _1.readStringValue() }) }
  }
  
//  fun <T> load(input: reader): NDArray<T> {
//    val shape = Dimension(input.readList())
//    val size = shape.numElements()
//    val type = input.readIntValue()
//    val buf = loadSwtich(type, input, size)
//    return NDArray(shape, buf) as NDArray<T>
//  }
  
  fun <T> save(output: writer, obj: NDArray<T>) {
    output.write(obj.shape.elements)
    val first = obj.first()
    saveSwtich(first as Any, output, obj)
  }
}