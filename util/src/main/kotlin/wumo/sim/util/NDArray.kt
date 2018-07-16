package wumo.sim.util

class NDArray<T>(shape: Dimension, val raw: Array<T>) {
  companion object {
    fun zeros(shape: Dimension): NDArray<Float> {
      return NDArray(shape, Array(shape.numElements()) { 0f })
    }
  }
  
  protected val stride: IntArray
  protected val dims: IntArray
  protected val numElements: Int
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
}