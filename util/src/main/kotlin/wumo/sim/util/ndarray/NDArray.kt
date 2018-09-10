@file:Suppress("UNCHECKED_CAST")

package wumo.sim.util.ndarray

import wumo.sim.util.*
import wumo.sim.util.ndarray.implementation.*

interface Buf<T> {
  operator fun get(offset: Int): T
  operator fun set(offset: Int, data: T)
  fun copy(): Buf<T>
  fun setFrom(other: Buf<T>) {
    for (i in 0 until size)
      this[i] = other[i]
  }
  
  fun slice(start: Int, end: Int): Buf<T>
  val size: Int
}

open class NDArray<T : Any>(val shape: Shape, val raw: Buf<T>, val dtype: Class<*> = raw[0]::class.java) : Iterable<T> {
  
  companion object {
    fun zeros(shape: Int): NDArray<Float> {
      return NDArray(Shape(shape), FloatArray(shape))
    }
    
    fun zeros(shape: Shape): NDArray<Float> {
      return NDArray(shape, FloatArray(shape.numElements()))
    }
    
    private val toNDArraySwitch = SwitchType<NDArray<*>>().apply {
      case<NDArray<*>> { it }
      case<Float> { NDArray(it) }
      case<Double> { NDArray(it) }
      case<Boolean> { NDArray(it) }
      case<Byte> { NDArray(it) }
      case<Short> { NDArray(it) }
      case<Int> { NDArray(it) }
      case<Long> { NDArray(it) }
      case<String> { NDArray(it) }
      case<FloatArray> { NDArray(it) }
      case<DoubleArray> { NDArray(it) }
      case<BooleanArray> { NDArray(it) }
      case<ByteArray> { NDArray(it) }
      case<ShortArray> { NDArray(it) }
      case<IntArray> { NDArray(it) }
      case<LongArray> { NDArray(it) }
      case<Array<Float>> { NDArray(it) }
      case<Array<Double>> { NDArray(it) }
      case<Array<Boolean>> { NDArray(it) }
      case<Array<Byte>> { NDArray(it) }
      case<Array<Short>> { NDArray(it) }
      case<Array<Int>> { NDArray(it) }
      case<Array<Long>> { NDArray(it) }
      case<Array<String>> { NDArray(it) }
      case<Array<NDArray<*>>> {
        combine(it.asIterable(), it.size)
      }
      caseIs<Array<*>> {
        val wrap = it.map { toNDArray(it!!) }
        combine(wrap, wrap.size)
      }
      caseIs<Collection<*>> {
        collectionSwitch(it.first()!!, it)
      }
    }
    private val collectionSwitch = SwitchType2<Collection<*>, NDArray<*>>().apply {
      case<Float> { NDArray((_2 as Collection<Float>).toFloatArray()) }
      case<Double> { NDArray((_2 as Collection<Double>).toDoubleArray()) }
      case<Boolean> { NDArray((_2 as Collection<Boolean>).toBooleanArray()) }
      case<Byte> { NDArray((_2 as Collection<Byte>).toByteArray()) }
      case<Short> { NDArray((_2 as Collection<Short>).toShortArray()) }
      case<Int> { NDArray((_2 as Collection<Int>).toIntArray()) }
      case<Long> { NDArray((_2 as Collection<Long>).toLongArray()) }
      case<String> { NDArray((_2 as Collection<String>).toTypedArray()) }
      case<NDArray<*>> {
        combine(_2 as Collection<NDArray<*>>, _2.size)
      }
      caseElse {
        val wrap = _2.map { toNDArray(it!!) }
        combine(wrap, wrap.size)
      }
    }
    
    private fun combine(c: Iterable<NDArray<*>>, size: Int): NDArray<*> {
      val first = c.first()
      val shape = Shape(size,first.shape)
      val firstElement = first.first()
      val buf = collectionNDArraySwitch(firstElement, shape.numElements()) as Buf<Any>
      
      var i = 0
      for (ndarray in c)
        for (element in ndarray) {
          buf[i++] = element
        }
      return NDArray(shape, buf, firstElement::class.java)
    }
    
    private val collectionNDArraySwitch = SwitchType2<Int, Buf<*>>().apply {
      case<Float> { FloatArrayBuf(FloatArray(_2)) }
      case<Double> { DoubleArrayBuf(DoubleArray(_2)) }
      case<Boolean> { BooleanArrayBuf(BooleanArray(_2)) }
      case<Byte> { ByteArrayBuf(ByteArray(_2)) }
      case<Short> { ShortArrayBuf(ShortArray(_2)) }
      case<Int> { IntArrayBuf(IntArray(_2)) }
      case<Long> { LongArrayBuf(LongArray(_2)) }
      case<String> { ArrayBuf(Array(_2) { "" }) }
    }
    
    fun toNDArray(value: Any): NDArray<*> = toNDArraySwitch(value)
    
    operator fun invoke(value: Float) = invoke(scalarDimension, f(value))
    operator fun invoke(value: Double) = invoke(scalarDimension, d(value))
    operator fun invoke(value: Boolean) = invoke(scalarDimension, B(value))
    operator fun invoke(value: Byte) = invoke(scalarDimension, b(value))
    operator fun invoke(value: Short) = invoke(scalarDimension, s(value))
    operator fun invoke(value: Int) = invoke(scalarDimension, i(value))
    operator fun invoke(value: Long) = invoke(scalarDimension, l(value))
    operator fun invoke(value: String) = invoke(scalarDimension, a(value))
    
    operator fun invoke(value: FloatArray) = NDArray(Shape(value.size), value)
    operator fun invoke(value: Array<Float>) = NDArray(Shape(value.size), value.toFloatArray())
    operator fun invoke(value: DoubleArray) = NDArray(Shape(value.size), value)
    operator fun invoke(value: Array<Double>) = NDArray(Shape(value.size), value.toDoubleArray())
    operator fun invoke(value: BooleanArray) = NDArray(Shape(value.size), value)
    operator fun invoke(value: Array<Boolean>) = NDArray(Shape(value.size), value.toBooleanArray())
    operator fun invoke(value: ByteArray) = NDArray(Shape(value.size), value)
    operator fun invoke(value: Array<Byte>) = NDArray(Shape(value.size), value.toByteArray())
    operator fun invoke(value: ShortArray) = NDArray(Shape(value.size), value)
    operator fun invoke(value: Array<Short>) = NDArray(Shape(value.size), value.toShortArray())
    operator fun invoke(value: IntArray) = NDArray(Shape(value.size), value)
    operator fun invoke(value: Array<Int>) = NDArray(Shape(value.size), value.toIntArray())
    operator fun invoke(value: LongArray) = NDArray(Shape(value.size), value)
    operator fun invoke(value: Array<Long>) = NDArray(Shape(value.size), value.toLongArray())
    operator fun invoke(value: Array<String>) = NDArray(Shape(value.size), value)
    
    operator fun invoke(shape: Shape, value: FloatArray) = NDArray(shape, FloatArrayBuf(value), Float::class.java)
    operator fun invoke(shape: Shape, value: DoubleArray) = NDArray(shape, DoubleArrayBuf(value), Double::class.java)
    operator fun invoke(shape: Shape, value: BooleanArray) = NDArray(shape, BooleanArrayBuf(value), Boolean::class.java)
    operator fun invoke(shape: Shape, value: ByteArray) = NDArray(shape, ByteArrayBuf(value), Byte::class.java)
    operator fun invoke(shape: Shape, value: ShortArray) = NDArray(shape, ShortArrayBuf(value), Short::class.java)
    operator fun invoke(shape: Shape, value: IntArray) = NDArray(shape, IntArrayBuf(value), Int::class.java)
    operator fun invoke(shape: Shape, value: LongArray) = NDArray(shape, LongArrayBuf(value), Long::class.java)
    operator fun invoke(shape: Shape, value: Array<String>) = NDArray(shape, ArrayBuf(value), String::class.java)
    
    operator fun invoke(shape: Shape, initvalue: Float) = NDArray(shape, FloatArray(shape.numElements()) { initvalue })
    operator fun invoke(shape: Shape, initvalue: Double) = NDArray(shape, DoubleArray(shape.numElements()) { initvalue })
    operator fun invoke(shape: Shape, initvalue: Boolean) = NDArray(shape, BooleanArray(shape.numElements()) { initvalue })
    operator fun invoke(shape: Shape, initvalue: Byte) = NDArray(shape, ByteArray(shape.numElements()) { initvalue })
    operator fun invoke(shape: Shape, initvalue: Short) = NDArray(shape, ShortArray(shape.numElements()) { initvalue })
    operator fun invoke(shape: Shape, initvalue: Int) = NDArray(shape, IntArray(shape.numElements()) { initvalue })
    operator fun invoke(shape: Shape, initvalue: Long) = NDArray(shape, LongArray(shape.numElements()) { initvalue })
    operator fun invoke(shape: Shape, initvalue: String) = NDArray(shape, Array(shape.numElements()) { initvalue })
  }
  
  private val stride: IntArray
  private val dims: IntArray
  /**number of elements*/
  val size: Int
  val numDims = shape.rank
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
  
  private inline fun <U> get_set(vararg idx: Int, op: (Int) -> U): U {
    val offset = if (idx.isEmpty()) 0L
    else {
      require(idx.size == dims.size)
      var offset = 0L
      for ((i, value) in idx.withIndex()) {
        require(value in 0 until dims[i]) { "dim($i)=$value is out of range[0,${dims[i]})" }
        offset += value * stride[i]
      }
      offset
    }
    return op(offset.toInt())
  }
  
  operator fun get(vararg idx: Int) = get_set(*idx) { raw[it] }
  
  operator fun set(vararg idx: Int, data: T) = get_set(*idx) {
    raw[it] = data
  }
  
  operator fun invoke(vararg idx: Int): NDArray<T> {
    var offset = 0
    for ((i, value) in idx.withIndex()) {
      offset += value * stride[i]
    }
    val size = stride[idx.size - 1]
    return NDArray(shape.slice(idx.size), raw.slice(offset, offset + size), dtype)
  }
  
  fun rawSet(idx: Int, data: T) {
    raw[idx] = data
  }
  
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
      if (this::element.isInitialized)
        element._2 = value
      else
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
  
  fun copy() = NDArray(shape, raw.copy(), dtype)
  fun setFrom(other: NDArray<T>) {
    raw.setFrom(other.raw)
  }
  
  fun reshape(newShape: Shape): NDArray<T> =
      NDArray(newShape, raw.copy(), dtype)
  
  operator fun component1() = raw[0]
  operator fun component2() = raw[1]
  operator fun component3() = raw[2]
  operator fun component4() = raw[3]
  operator fun component5() = raw[4]
  operator fun component6() = raw[5]
}