package wumo.sim.util.ndarray

import wumo.sim.util.*
import wumo.sim.util.ndarray.implementation.*

interface Buf<T> {
  operator fun get(offset: Int): T
  operator fun set(offset: Int, data: T)
}

open class NDArray<T>(shape: Dimension, val raw: Buf<T>) : Iterable<T> {
  companion object {
    fun zeros(shape: Dimension): NDArray<Float> {
      return NDArray(shape, FloatArrayBuf(FloatArray(shape.numElements())))
    }
    
    operator fun invoke(value: Float) = invoke(scalarDimension, f(value))
    operator fun invoke(value: Double) = invoke(scalarDimension, d(value))
    operator fun invoke(value: Boolean) = invoke(scalarDimension, B(value))
    operator fun invoke(value: Byte) = invoke(scalarDimension, b(value))
    operator fun invoke(value: Short) = invoke(scalarDimension, s(value))
    operator fun invoke(value: Int) = invoke(scalarDimension, i(value))
    operator fun invoke(value: Long) = invoke(scalarDimension, l(value))
    operator fun invoke(value: String) = invoke(scalarDimension, a(value))
    
    operator fun invoke(shape: Dimension, value: FloatArray) = NDArray(shape, FloatArrayBuf(value))
    operator fun invoke(shape: Dimension, value: DoubleArray) = NDArray(shape, DoubleArrayBuf(value))
    operator fun invoke(shape: Dimension, value: BooleanArray) = NDArray(shape, BooleanArrayBuf(value))
    operator fun invoke(shape: Dimension, value: ByteArray) = NDArray(shape, ByteArrayBuf(value))
    operator fun invoke(shape: Dimension, value: ShortArray) = NDArray(shape, ShortArrayBuf(value))
    operator fun invoke(shape: Dimension, value: IntArray) = NDArray(shape, IntArrayBuf(value))
    operator fun invoke(shape: Dimension, value: LongArray) = NDArray(shape, LongArrayBuf(value))
    operator fun invoke(shape: Dimension, value: Array<String>) = NDArray(shape, ArrayBuf(value))
  }
  
  private val stride: IntArray
  private val dims: IntArray
  private val numElements: Int
  val numDims = shape.rank()
  
  init {
    stride = IntArray(numDims)
    dims = IntArray(numDims) { shape[it] }
    var n = 0
    if (dims.isNotEmpty()) {
      stride[stride.lastIndex] = 1
      for (a in stride.lastIndex - 1 downTo 0)
        stride[a] = dims[a + 1] * stride[a + 1]
      n = dims[0] * stride[0]
    }
    numElements = n
  }
  
  private inline fun <U> get_set(vararg idx: Int, op: (Int) -> U): U {
    require(idx.size == dims.size)
    var offset = 0L
    for ((i, value) in idx.withIndex()) {
      require(value in 0 until dims[i]) { "dim($i)=$value is out of range[0,${dims[i]})" }
      offset += value * stride[i]
    }
    return op(offset.toInt())
  }
  
  operator fun get(vararg idx: Int) = get_set(*idx) { raw[it] }
  
  operator fun set(vararg idx: Int, data: T) = get_set(*idx) {
    raw[it] = data
  }
  
  override fun iterator() = object : Iterator<T> {
    var a = 0
    override fun hasNext() = a < numElements
    
    override fun next() = raw[a++]
  }
  
  fun withIndex() = object : Iterator<tuple2<IntArray, T>> {
    val idx = IntArray(dims.size).apply { this[this.lastIndex] = -1 }
    var a = 0
    lateinit var element: tuple2<IntArray, T>
    override fun hasNext() = a < numElements
    
    override fun next(): tuple2<IntArray, T> {
      for (_idx in dims.lastIndex downTo 0) {
        idx[_idx]++
        if (idx[_idx] < dims[_idx])
          break
        idx[_idx] = 0
      }
      val value = raw[a++]
      if (this::element.isInitialized)
        element._2 = value
      else
        element = tuple2(idx, value)
      return element
    }
  }
  
  @Suppress("NAME_SHADOWING")
  private fun printTensor(extraPadding: Int, dim: Int, offset: Int, sb: StringBuilder) {
    val padding = StringBuilder().let { s ->
      repeat(dim + 1 + extraPadding) { s.append(' ') }
      s.toString()
    }
    var offset = offset
    val isVector = dim == dims.lastIndex - 1
    sb.append('[')
    for (i in 0 until dims[dim]) {
      if (i != 0) sb.append(padding)
      if (isVector)
        printVector(offset, stride[dim].toInt(), sb)
      else
        printTensor(extraPadding, dim + 1, offset, sb)
      if (i != dims[dim] - 1)
        sb.append(",\n")
      offset += stride[dim].toInt()
    }
    sb.append(']')
  }
  
  private fun printVector(offset: Int, size: Int, sb: StringBuilder) {
    sb.append('[')
    for (j in 0 until size) {
      sb.append(raw[offset + j])
      if (j != size - 1)
        sb.append(',')
    }
    sb.append(']')
  }
  
  fun toString(padding: Int): String {
    val sb = StringBuilder()
    when (dims.size) {
      0 -> return get().toString()
      1 -> printVector(0, dims[0].toInt(), sb)
      else -> printTensor(padding, 0, 0, sb)
    }
    return sb.toString()
  }
  
  override fun toString(): String {
    val sb = StringBuilder()
    when (dims.size) {
      0 -> return get().toString()
      1 -> printVector(0, dims[0], sb)
      else -> printTensor(0, 0, 0, sb)
    }
    return sb.toString()
  }
}