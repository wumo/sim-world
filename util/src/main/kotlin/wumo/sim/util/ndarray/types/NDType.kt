package wumo.sim.util.ndarray.types

import org.bytedeco.javacpp.BytePointer
import wumo.sim.util.NONE
import wumo.sim.util.SwitchOnClass
import wumo.sim.util.ndarray.BytePointerBuf

sealed class NDType<KotlinType : Any> {
  val name: String = javaClass.simpleName
  abstract val byteSize: Int
  abstract val kotlinType: Class<KotlinType>
  
  abstract fun zero(): KotlinType
  abstract fun one(): KotlinType
  
  abstract fun <R : Any> cast(value: R): KotlinType
  
  open fun makeBuf(size: Int, init: (Int) -> KotlinType): BytePointerBuf<KotlinType> =
      BytePointerBuf(size, this, init)
  
  abstract fun put(buf: BytePointer, offset: Long, data: KotlinType)
  
  abstract fun get(buf: BytePointer, offset: Long): KotlinType
  
  override fun equals(other: Any?): Boolean = (other is NDType<*>) && (kotlinType == other.kotlinType)
  override fun hashCode(): Int = kotlinType.hashCode()
  override fun toString(): String = name
}

private val JavaClsToNDType = SwitchOnClass<NDType<*>>().apply {
  case<Boolean> { NDBool }
  case<Byte> { NDByte }
  case<Short> { NDShort }
  case<Int> { NDInt }
  case<Long> { NDLong }
  case<Float> { NDFloat }
  case<Double> { NDDouble }
  case<String> { NDString }
}

fun <T : Any> Class<T>.NDType(): NDType<T> = JavaClsToNDType(this) as NDType<T>
fun <T : Any> T.NDType(): NDType<T> = JavaClsToNDType(javaClass) as NDType<T>

object NDBool : NDType<Boolean>() {
  override val byteSize = 1
  override val kotlinType = Boolean::class.java
  
  override fun zero() = false
  
  override fun one() = true
  
  override fun <R : Any> cast(value: R): Boolean =
      when (value) {
        is Boolean -> value
        else -> NONE()
      }
  
  override fun put(buf: BytePointer, offset: Long, data: Boolean) {
    buf.putBool(offset, data)
  }
  
  override fun get(buf: BytePointer, offset: Long): Boolean =
      buf.getBool(offset)
}

object NDByte : NDType<Byte>() {
  override val byteSize = Byte.SIZE_BYTES
  override val kotlinType = Byte::class.java
  
  override fun zero(): Byte = 0
  
  override fun one(): Byte = 1
  
  override fun <R : Any> cast(value: R): Byte =
      when (value) {
        is Number -> value.toByte()
        is Boolean -> if (value) 1 else 0
        else -> NONE()
      }
  
  override fun put(buf: BytePointer, offset: Long, data: Byte) {
    buf.put(offset, data)
  }
  
  override fun get(buf: BytePointer, offset: Long): Byte =
      buf.get(offset)
}

object NDShort : NDType<Short>() {
  override val byteSize = Short.SIZE_BYTES
  override val kotlinType = Short::class.java
  
  override fun zero(): Short = 0
  
  override fun one(): Short = 1
  
  override fun <R : Any> cast(value: R): Short = when (value) {
    is Number -> value.toShort()
    is Boolean -> if (value) 1 else 0
    else -> NONE()
  }
  
  override fun put(buf: BytePointer, offset: Long, data: Short) {
    buf.putShort(offset, data)
  }
  
  override fun get(buf: BytePointer, offset: Long): Short =
      buf.getShort(offset)
}

object NDInt : NDType<Int>() {
  override val byteSize = Int.SIZE_BYTES
  override val kotlinType = Int::class.java
  
  override fun zero(): Int = 0
  
  override fun one(): Int = 1
  
  override fun <R : Any> cast(value: R): Int = when (value) {
    is Number -> value.toInt()
    is Boolean -> if (value) 1 else 0
    else -> NONE()
  }
  
  override fun put(buf: BytePointer, offset: Long, data: Int) {
    buf.putInt(offset, data)
  }
  
  override fun get(buf: BytePointer, offset: Long): Int =
      buf.getInt(offset)
}

object NDLong : NDType<Long>() {
  override val byteSize = Long.SIZE_BYTES
  override val kotlinType = Long::class.java
  
  override fun zero(): Long = 0
  
  override fun one(): Long = 1
  
  override fun <R : Any> cast(value: R): Long = when (value) {
    is Number -> value.toLong()
    is Boolean -> if (value) 1 else 0
    else -> NONE()
  }
  
  override fun put(buf: BytePointer, offset: Long, data: Long) {
    buf.putLong(offset, data)
  }
  
  override fun get(buf: BytePointer, offset: Long): Long =
      buf.getLong(offset)
}

object NDFloat : NDType<Float>() {
  override val byteSize = 4
  override val kotlinType = Float::class.java
  
  override fun zero(): Float = 0f
  
  override fun one(): Float = 1f
  
  override fun <R : Any> cast(value: R): Float = when (value) {
    is Number -> value.toFloat()
    is Boolean -> if (value) 1f else 0f
    else -> NONE()
  }
  
  override fun put(buf: BytePointer, offset: Long, data: Float) {
    buf.putFloat(offset, data)
  }
  
  override fun get(buf: BytePointer, offset: Long): Float =
      buf.getFloat(offset)
}

object NDDouble : NDType<Double>() {
  override val byteSize = 8
  override val kotlinType = Double::class.java
  
  override fun zero(): Double = 0.0
  
  override fun one(): Double = 1.0
  
  override fun <R : Any> cast(value: R): Double = when (value) {
    is Number -> value.toDouble()
    is Boolean -> if (value) 1.0 else 0.0
    else -> NONE()
  }
  
  override fun put(buf: BytePointer, offset: Long, data: Double) {
    buf.putDouble(offset, data)
  }
  
  override fun get(buf: BytePointer, offset: Long): Double =
      buf.getDouble(offset)
}

object NDString : NDType<String>() {
  override val byteSize = -1
  override val kotlinType = String::class.java
  
  override fun zero(): String = ""
  
  override fun one(): String = NONE()
  
  override fun <R : Any> cast(value: R): String = value.toString()
  
  override fun makeBuf(size: Int, init: (Int) -> String): BytePointerBuf<String> =
      TODO()
  
  override fun put(buf: BytePointer, offset: Long, data: String) {
    TODO("not implemented")
  }
  
  override fun get(buf: BytePointer, offset: Long): String {
    TODO("not implemented")
  }
}