package wumo.sim.util.ndarray

import wumo.sim.util.Shape

fun <T : Any> concatenate(array: List<NDArray<T>>, axis: Int = 0): NDArray<T> {
  val iter = array.iterator()
  val first = iter.next()
  val finalShapeDims = first.shape.asIntArray()!!.copyOf()
  val offset = IntArray(array.size)
  var finalDim = finalShapeDims[axis]
  var i = 0
  offset[i++] = finalShapeDims[axis]
  while (iter.hasNext()) {
    val shape = iter.next().shape
    offset[i++] = shape[axis]
    finalDim += shape[axis]
  }
  finalShapeDims[axis] = finalDim
  for (i in 0 until offset.size)
    if (i > 0)
      offset[i] += offset[i - 1]
  val finalShape = Shape(finalShapeDims)
  val idx = IntArray(finalShape.rank)
  var j = 0
  return NDArray(finalShape, first.dtype.makeBuf(finalShape.numElements()) {
    val k = idx[axis]
    var min = 0
    outer@ while (true) {
      min = if (j > 0) offset[j - 1] else 0
      val max = offset[j]
      when (k) {
        in min until max -> break@outer
        max -> {
          j++
          require(j < array.size)
        }
        0 -> j = 0
        else -> error("current idx:$k, current j:$j, min:$min, max:$max")
      }
    }
    idx[axis] = k - min
    val element = array[j].get(*idx)
    element.apply {
      idx[axis] = k
      idx.advance(finalShape)
    }
  }, first.dtype)
}