@file:Suppress("UNCHECKED_CAST")

package wumo.sim.util.ndarray

import org.bytedeco.javacpp.Pointer.memcpy
import wumo.sim.buf
import wumo.sim.util.*
import wumo.sim.util.ndarray.types.*

fun <T : Any> Any.toNDArray(shape: Shape? = null): NDArray<T> = NDArray.toNDArray(this, shape) as NDArray<T>

fun IntArray.advance(shape: Shape) {
  var i = lastIndex
  do {
    this[i]++
    if (this[i] < shape[i]) break
    this[i] = 0
    i--
  } while (i >= 0)
}

val castSwitch = SwitchOnClass1<Number, Any>().apply {
  case<Byte> { it.toByte() }
  case<Int> { it.toInt() }
  case<Short> { it.toShort() }
  case<Long> { it.toLong() }
  case<Float> { it.toFloat() }
  case<Double> { it.toDouble() }
}

fun <R : Number, T : Number> R.cast(dataType: Class<T>): T =
    castSwitch(dataType, this as Number) as T

open class NDArray<T : Any>(val shape: Shape, val raw: BytePointerBuf<T>) : Iterable<T> {
  
  companion object {
    init {
      buf.buf_init()
    }
    
    fun zeros(shape: Int): NDArray<Float> {
      return NDArray(Shape(shape), 0f)
    }
    
    fun zeros(shape: Shape): NDArray<Float> {
      return NDArray(shape, 0f)
    }
    
    private val toBufSwitch = SwitchType<Pair<BytePointerBuf<*>, Shape>>().apply {
      case<NDArray<*>> { it.raw to it.shape }
      case<Float> { v -> BytePointerBuf(1, NDFloat) { v } to scalarDimension }
      case<Double> { v -> BytePointerBuf(1, NDDouble) { v } to scalarDimension }
      case<Boolean> { v -> BytePointerBuf(1, NDBool) { v } to scalarDimension }
      case<Byte> { v -> BytePointerBuf(1, NDByte) { v } to scalarDimension }
      case<Short> { v -> BytePointerBuf(1, NDShort) { v } to scalarDimension }
      case<Int> { v -> BytePointerBuf(1, NDInt) { v } to scalarDimension }
      case<Long> { v -> BytePointerBuf(1, NDLong) { v } to scalarDimension }
//      case<String> { v -> BytePointerBuf(1, NDString) { v } to scalarDimension }
      case<FloatArray> { v -> BytePointerBuf(v.size, NDFloat) { v[it] } to Shape(v.size) }
      case<DoubleArray> { v -> BytePointerBuf(v.size, NDDouble) { v[it] } to Shape(v.size) }
      case<BooleanArray> { v -> BytePointerBuf(v.size, NDBool) { v[it] } to Shape(v.size) }
      case<ByteArray> { v -> BytePointerBuf(v.size, NDByte) { v[it] } to Shape(v.size) }
      case<ShortArray> { v -> BytePointerBuf(v.size, NDShort) { v[it] } to Shape(v.size) }
      case<IntArray> { v -> BytePointerBuf(v.size, NDInt) { v[it] } to Shape(v.size) }
      case<LongArray> { v -> BytePointerBuf(v.size, NDLong) { v[it] } to Shape(v.size) }
      case<Array<Float>> { v -> BytePointerBuf(v.size, NDFloat) { v[it] } to Shape(v.size) }
      case<Array<Double>> { v -> BytePointerBuf(v.size, NDDouble) { v[it] } to Shape(v.size) }
      case<Array<Boolean>> { v -> BytePointerBuf(v.size, NDBool) { v[it] } to Shape(v.size) }
      case<Array<Byte>> { v -> BytePointerBuf(v.size, NDByte) { v[it] } to Shape(v.size) }
      case<Array<Short>> { v -> BytePointerBuf(v.size, NDShort) { v[it] } to Shape(v.size) }
      case<Array<Int>> { v -> BytePointerBuf(v.size, NDInt) { v[it] } to Shape(v.size) }
      case<Array<Long>> { v -> BytePointerBuf(v.size, NDLong) { v[it] } to Shape(v.size) }
//      case<Array<String>> { v -> BytePointerBuf(v.size, NDString) { v[it] } to Shape(v.size) }
      
      case<Array<NDArray<*>>> {
        expand(it.asIterable(), it.size)
      }
      caseIs<Array<*>> {
        val wrap = it.map { toNDArray(it!!) }
        expand(wrap, wrap.size)
      }
      caseIs<Collection<*>> {
        collectionSwitch(it.first()!!, it)
      }
    }
    private val collectionSwitch = SwitchType2<Collection<*>, Pair<BytePointerBuf<*>, Shape>>().apply {
      case<Float> {
        val iter = (_2 as Collection<Float>).iterator()
        BytePointerBuf(_2.size, NDFloat) { iter.next() } to Shape(_2.size)
      }
      case<Double> {
        val iter = (_2 as Collection<Double>).iterator()
        BytePointerBuf(_2.size, NDDouble) { iter.next() } to Shape(_2.size)
      }
      case<Boolean> {
        val iter = (_2 as Collection<Boolean>).iterator()
        BytePointerBuf(_2.size, NDBool) { iter.next() } to Shape(_2.size)
      }
      case<Byte> {
        val iter = (_2 as Collection<Byte>).iterator()
        BytePointerBuf(_2.size, NDByte) { iter.next() } to Shape(_2.size)
      }
      case<Short> {
        val iter = (_2 as Collection<Short>).iterator()
        BytePointerBuf(_2.size, NDShort) { iter.next() } to Shape(_2.size)
      }
      case<Int> {
        val iter = (_2 as Collection<Int>).iterator()
        BytePointerBuf(_2.size, NDInt) { iter.next() } to Shape(_2.size)
      }
      case<Long> {
        val iter = (_2 as Collection<Long>).iterator()
        BytePointerBuf(_2.size, NDLong) { iter.next() } to Shape(_2.size)
      }
//      case<String> {
//        val iter = (_2 as Collection<String>).iterator()
//        BytePointerBuf(_2.size, NDString) { iter.next() } to Shape(_2.size)
//      }
      case<NDArray<*>> {
        expand(_2 as Collection<NDArray<*>>, _2.size)
      }
      caseElse {
        val wrap = _2.map { toNDArray(it!!) }
        expand(wrap, wrap.size)
      }
    }
    
    private fun expand(c: Iterable<NDArray<*>>, size: Int): Pair<BytePointerBuf<*>, Shape> {
      val first = c.first()
      val shape = Shape(size, first.shape)
      val firstElement = first.first()
      val buf = collectionNDArraySwitch(firstElement, shape.numElements()) as BytePointerBuf<Any>
      
      val ptr = buf.ptr
      var i = 0L
      for (ndarray in c) {
        val src = ndarray.raw.ptr
        ptr.position(i)
        memcpy(ptr, src, src.capacity())
        i += src.capacity()
      }
      ptr.position(0)
      return buf to shape
    }
    
    private val collectionNDArraySwitch = SwitchType2<Int, BytePointerBuf<*>>().apply {
      case<Float> { BytePointerBuf(_2, NDFloat) }
      case<Double> { BytePointerBuf(_2, NDDouble) }
      case<Boolean> { BytePointerBuf(_2, NDBool) }
      case<Byte> { BytePointerBuf(_2, NDByte) }
      case<Short> { BytePointerBuf(_2, NDShort) }
      case<Int> { BytePointerBuf(_2, NDInt) }
      case<Long> { BytePointerBuf(_2, NDLong) }
//      case<String> { BytePointerBuf(_2, NDString) }
    }
    
    fun toNDArray(value: Any, shape: Shape? = null): NDArray<*> {
      val (buf, inferredShape) = toBufSwitch(value)
      return NDArray(shape ?: inferredShape, buf)
    }
    
    operator fun invoke(value: Float) = invoke(scalarDimension, NDFloat) { value }
    operator fun invoke(value: Double) = invoke(scalarDimension, NDDouble) { value }
    operator fun invoke(value: Boolean) = invoke(scalarDimension, NDBool) { value }
    operator fun invoke(value: Byte) = invoke(scalarDimension, NDByte) { value }
    operator fun invoke(value: Short) = invoke(scalarDimension, NDShort) { value }
    operator fun invoke(value: Int) = invoke(scalarDimension, NDInt) { value }
    operator fun invoke(value: Long) = invoke(scalarDimension, NDLong) { value }
//    operator fun invoke(value: String) = invoke(scalarDimension, NDString) { value }
    
    operator fun invoke(value: FloatArray) = invoke(Shape(value.size), NDFloat) { value[it] }
    operator fun invoke(value: Array<Float>) = invoke(Shape(value.size), NDFloat) { value[it] }
    operator fun invoke(value: DoubleArray) = invoke(Shape(value.size), NDDouble) { value[it] }
    operator fun invoke(value: Array<Double>) = invoke(Shape(value.size), NDDouble) { value[it] }
    operator fun invoke(value: BooleanArray) = invoke(Shape(value.size), NDBool) { value[it] }
    operator fun invoke(value: Array<Boolean>) = invoke(Shape(value.size), NDBool) { value[it] }
    operator fun invoke(value: ByteArray) = invoke(Shape(value.size), NDByte) { value[it] }
    operator fun invoke(value: Array<Byte>) = invoke(Shape(value.size), NDByte) { value[it] }
    operator fun invoke(value: ShortArray) = invoke(Shape(value.size), NDShort) { value[it] }
    operator fun invoke(value: Array<Short>) = invoke(Shape(value.size), NDShort) { value[it] }
    operator fun invoke(value: IntArray) = invoke(Shape(value.size), NDInt) { value[it] }
    operator fun invoke(value: Array<Int>) = invoke(Shape(value.size), NDInt) { value[it] }
    operator fun invoke(value: LongArray) = invoke(Shape(value.size), NDLong) { value[it] }
    operator fun invoke(value: Array<Long>) = invoke(Shape(value.size), NDLong) { value[it] }
//    operator fun invoke(value: Array<String>) = invoke(Shape(value.size), value)
    
    operator fun invoke(shape: Shape, value: FloatArray) = invoke(shape, NDFloat) { value[it] }
    operator fun invoke(shape: Shape, value: DoubleArray) = invoke(shape, NDDouble) { value[it] }
    operator fun invoke(shape: Shape, value: BooleanArray) = invoke(shape, NDBool) { value[it] }
    operator fun invoke(shape: Shape, value: ByteArray) = invoke(shape, NDByte) { value[it] }
    operator fun invoke(shape: Shape, value: ShortArray) = invoke(shape, NDShort) { value[it] }
    operator fun invoke(shape: Shape, value: IntArray) = invoke(shape, NDInt) { value[it] }
    operator fun invoke(shape: Shape, value: LongArray) = invoke(shape, NDLong) { value[it] }
//    operator fun invoke(shape: Shape, value: Array<String>) = NDArray(shape, ArrayBuf(value), NDString)
    
    operator fun invoke(shape: Shape, initvalue: Float) = invoke(shape, NDFloat) { initvalue }
    operator fun invoke(shape: Shape, initvalue: Double) = invoke(shape, NDDouble) { initvalue }
    operator fun invoke(shape: Shape, initvalue: Boolean) = invoke(shape, NDBool) { initvalue }
    operator fun invoke(shape: Shape, initvalue: Byte) = invoke(shape, NDByte) { initvalue }
    operator fun invoke(shape: Shape, initvalue: Short) = invoke(shape, NDShort) { initvalue }
    operator fun invoke(shape: Shape, initvalue: Int) = invoke(shape, NDInt) { initvalue }
    operator fun invoke(shape: Shape, initvalue: Long) = invoke(shape, NDLong) { initvalue }
//    operator fun invoke(shape: Shape, initvalue: String) = invoke(shape, NDString) { initvalue }
    
    inline operator fun <T : Any> invoke(shape: Shape, dtype: NDType<T>, init: (Int) -> T): NDArray<T> =
        NDArray(shape, BytePointerBuf(shape.numElements(), dtype, init))
    
    inline fun <reified T : Any> from(shape: Shape, initvalue: (IntArray) -> T): NDArray<T> {
      val idx = IntArray(shape.rank)
      return Array(shape.numElements()) {
        initvalue(idx).apply {
          idx.advance(shape)
        }
      }.toNDArray(shape)
    }
  }
  
  val stride: IntArray
  private val dims: IntArray
  /**number of elements*/
  val size: Int
  val numDims = shape.rank
  val dtype = raw.ndType
  
  init {
    stride = IntArray(numDims)
    dims = shape.asIntArray()!!
    if (dims.isNotEmpty()) {
      stride[stride.lastIndex] = 1
      for (a in stride.lastIndex - 1 downTo 0)
        stride[a] = dims[a + 1] * stride[a + 1]
    }
    size = raw.size
  }
  
  val numElements = size
  inline val isScalar
    get() = shape.isScalar
  
  val scalar: T
    get() {
      require(isScalar)
      return raw[0]
    }
  
  private fun idxToOffset(vararg idx: Int): Int {
    val offset = if (idx.isEmpty()) 0L
    else {
      var offset = 0L
      for ((i, value) in idx.withIndex()) {
        require(value in 0 until dims[i]) { "dim($i)=$value is out of range[0,${dims[i]})" }
        offset += value * stride[i]
      }
      offset
    }
    return offset.toInt()
  }
  
  private inline fun <U> get_set(vararg idx: Int, op: (Int) -> U): U {
    require(idx.size == dims.size)
    return op(idxToOffset(*idx))
  }
  
  operator fun get(vararg idx: Int) = get_set(*idx) { raw[it] }
  
  operator fun set(vararg idx: Int, data: T) = get_set(*idx) {
    raw[it] = data
  }
  
  operator fun set(vararg idx: Int, data: NDArray<T>) {
    val offset = idxToOffset(*idx)
    for ((i, v) in data.flatten())
      raw[offset + i] = v
  }
  
  operator fun invoke(vararg idx: Int): NDArray<T> {
    val offset = idxToOffset(*idx)
    val size = stride[idx.size - 1]
    return NDArray(shape.slice(idx.size), raw.slice(offset, offset + size))
  }
  
  fun rawSet(idx: Int, data: T) {
    raw[idx] = data
  }
  
  fun rawGet(idx: Int): T = raw[idx]
  
  override fun iterator() = object : Iterator<T> {
    var a = 0
    override fun hasNext() = a < size
    
    override fun next() = raw[a++]
  }
  
  fun flatten() = object : Iterator<t2<Int, T>> {
    var a = 0
    lateinit var element: t2<Int, T>
    override fun hasNext() = a < size
    
    override fun next(): t2<Int, T> {
      val value = raw[a]
      if (this::element.isInitialized) {
        element._1 = a
        element._2 = value
      } else
        element = t2(a, value)
      a++
      return element
    }
  }
  
  fun withIndex() = object : Iterator<t2<IntArray, T>> {
    val idx = IntArray(dims.size).apply { this[this.lastIndex] = -1 }
    var a = 0
    lateinit var element: t2<IntArray, T>
    override fun hasNext() = a < size
    
    override fun next(): t2<IntArray, T> {
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
        element = t2(idx, value)
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
      1 -> printVector(0, dims[0], sb)
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
  
  fun copy() = NDArray(shape, raw.copy())
  fun setFrom(other: NDArray<T>) {
    memcpy(raw.ptr, other.raw.ptr, raw.ptr.capacity())
  }
  
  fun reshape(newShape: Shape): NDArray<T> =
      NDArray(newShape, raw.copy())
  
  operator fun component1() = raw[0]
  operator fun component2() = raw[1]
  operator fun component3() = raw[2]
  operator fun component4() = raw[3]
  operator fun component5() = raw[4]
  operator fun component6() = raw[5]
  
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (javaClass != other?.javaClass) return false
    
    other as NDArray<*>
    
    if (shape != other.shape) return false
    if (raw != other.raw) return false
    return true
  }
  
  override fun hashCode(): Int {
    var result = shape.hashCode()
    result = 31 * result + raw.hashCode()
    return result
  }
}