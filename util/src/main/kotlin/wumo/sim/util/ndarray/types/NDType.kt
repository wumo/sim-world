package wumo.sim.util.ndarray.types

import wumo.sim.util.NONE
import wumo.sim.util.SwitchOnClass
import wumo.sim.util.SwitchType
import wumo.sim.util.ndarray.Buf
import wumo.sim.util.ndarray.implementation.*

sealed class NDType<KotlinType : Any> {
  val name: String = javaClass.simpleName
  abstract val byteSize: Int
  abstract val kotlinType: Class<KotlinType>
  
  abstract fun zero(): KotlinType
  abstract fun one(): KotlinType
  
  abstract fun <R> cast(value: R): KotlinType
  
  abstract fun makeBuf(size: Int, init: (Int) -> KotlinType): Buf<KotlinType>
  
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
  
  override fun <R> cast(value: R): Boolean =
      when (value) {
        is Boolean -> value
        else -> NONE()
      }
  
  override fun makeBuf(size: Int, init: (Int) -> Boolean): Buf<Boolean> =
      BooleanArrayBuf(BooleanArray(size, init))
}

object NDByte : NDType<Byte>() {
  override val byteSize = Byte.SIZE_BYTES
  override val kotlinType = Byte::class.java
  
  override fun zero(): Byte = 0
  
  override fun one(): Byte = 1
  
  override fun <R> cast(value: R): Byte = when (value) {
    is Boolean -> if (value) 1 else 0
    is Float -> value.toByte()
    is Double -> value.toByte()
    is Byte -> value.toByte()
    is Short -> value.toByte()
    is Int -> value.toByte()
    is Long -> value.toByte()
    else -> NONE()
  }
  
  override fun makeBuf(size: Int, init: (Int) -> Byte): Buf<Byte> =
      ByteArrayBuf(ByteArray(size, init))
}

object NDShort : NDType<Short>() {
  override val byteSize = Short.SIZE_BYTES
  override val kotlinType = Short::class.java
  
  override fun zero(): Short = 0
  
  override fun one(): Short = 1
  
  override fun <R> cast(value: R): Short = when (value) {
    is Boolean -> if (value) 1 else 0
    is Float -> value.toShort()
    is Double -> value.toShort()
    is Byte -> value.toShort()
    is Short -> value.toShort()
    is Int -> value.toShort()
    is Long -> value.toShort()
    else -> NONE()
  }
  
  override fun makeBuf(size: Int, init: (Int) -> Short): Buf<Short> =
      ShortArrayBuf(ShortArray(size, init))
}

object NDInt : NDType<Int>() {
  override val byteSize = Int.SIZE_BYTES
  override val kotlinType = Int::class.java
  
  override fun zero(): Int = 0
  
  override fun one(): Int = 1
  
  override fun <R> cast(value: R): Int = when (value) {
    is Boolean -> if (value) 1 else 0
    is Float -> value.toInt()
    is Double -> value.toInt()
    is Byte -> value.toInt()
    is Short -> value.toInt()
    is Int -> value.toInt()
    is Long -> value.toInt()
    else -> NONE()
  }
  
  override fun makeBuf(size: Int, init: (Int) -> Int): Buf<Int> =
      IntArrayBuf(IntArray(size, init))
}

object NDLong : NDType<Long>() {
  override val byteSize = Long.SIZE_BYTES
  override val kotlinType = Long::class.java
  
  override fun zero(): Long = 0
  
  override fun one(): Long = 1
  
  override fun <R> cast(value: R): Long = when (value) {
    is Boolean -> if (value) 1L else 0L
    is Float -> value.toLong()
    is Double -> value.toLong()
    is Byte -> value.toLong()
    is Short -> value.toLong()
    is Int -> value.toLong()
    is Long -> value.toLong()
    else -> NONE()
  }
  
  override fun makeBuf(size: Int, init: (Int) -> Long): Buf<Long> =
      LongArrayBuf(LongArray(size, init))
}

object NDFloat : NDType<Float>() {
  override val byteSize = 4
  override val kotlinType = Float::class.java
  
  override fun zero(): Float = 0f
  
  override fun one(): Float = 1f
  
  override fun <R> cast(value: R): Float = when (value) {
    is Boolean -> if (value) 1.0f else 0.0f
    is Float -> value.toFloat()
    is Double -> value.toFloat()
    is Byte -> value.toFloat()
    is Short -> value.toFloat()
    is Int -> value.toFloat()
    is Long -> value.toFloat()
    else -> NONE()
  }
  
  override fun makeBuf(size: Int, init: (Int) -> Float): Buf<Float> =
      FloatArrayBuf(FloatArray(size, init))
}

object NDDouble : NDType<Double>() {
  override val byteSize = 8
  override val kotlinType = Double::class.java
  
  override fun zero(): Double = 0.0
  
  override fun one(): Double = 1.0
  
  override fun <R> cast(value: R): Double = when (value) {
    is Boolean -> if (value) 1.0 else 0.0
    is Float -> value.toDouble()
    is Double -> value.toDouble()
    is Byte -> value.toDouble()
    is Short -> value.toDouble()
    is Int -> value.toDouble()
    is Long -> value.toDouble()
    else -> NONE()
  }
  
  override fun makeBuf(size: Int, init: (Int) -> Double): Buf<Double> =
      DoubleArrayBuf(DoubleArray(size, init))
}

object NDString : NDType<String>() {
  override val byteSize = -1
  override val kotlinType = String::class.java
  
  override fun zero(): String = ""
  
  override fun one(): String = NONE()
  
  override fun <R> cast(value: R): String = value.toString()
  
  override fun makeBuf(size: Int, init: (Int) -> String): Buf<String> =
      ArrayBuf(Array(size, init))
}