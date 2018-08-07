@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.util

infix fun Int.x(a: Int) = dim(this, a)
infix fun Shape.x(a: Int) = dim(*asIntArray()!!, a)
infix fun Int.x(shape: Shape) = dim(this, *shape.asIntArray()!!)

inline fun dim(vararg d: Int) = Shape(d)

val scalarDimension = Shape()

/** Represents the shape of a tensor computed by an op.
 *
 * A [Shape] represents a possibly-partial shape specification for an op output. It may be one of the following:
 *
 *  - Fully known shape: It has a known number of dimensions and a known size for each dimension.
 *  - Partially known shape: It has a known number of dimensions and an unknown size for one or more dimension.
 *  - Unknown shape: It has an unknown number of dimensions and an unknown size for all its dimensions.
 *
 * Unknown dimensions are represented as having a size of `-1` and two shapes are considered equal only if they are
 * fully known and all their dimension sizes match.
 *
 * If a tensor is produced by an op of type `"Foo"`, its shape may be inferred if there is a registered shape function
 * for `"Foo"`. See [https://www.tensorflow.org/extend/adding_an_op#shape_functions_in_c Shape Functions in C++] for
 * details on shape functions and how to register them.
 *
 * @see [tensorflow/python/framework/tensor_shape.py/TensorShape]
 */
class Shape(private val dims: IntArray? = null) : Iterable<Int> {
  constructor(elements: LongArray) : this(IntArray(elements.size) { elements[it].toInt() })
  
  fun asLongArray() =
      LongArray(dims!!.size) { dims[it].toLong() }
  
  fun asIntArray() = dims
  
  fun rank() = dims?.size ?: -1
  
  val is_fully_defined: Boolean
    get() = dims?.all { it >= 0 } ?: false
  
  fun numElements() =
      if (!is_fully_defined) -1
      else {
        var size = 1
        dims!!.forEach { size *= it }
        size
      }
  
  operator fun get(idx: Int) =
      if (idx < 0)
        dims!![dims.size + idx]
      else
        dims!![idx]
  
  operator fun get(idx: IntRange) = run {
    val size = (idx.endInclusive - idx.start) / idx.step
    val iter = idx.iterator()
    Shape(IntArray(size) {
      this[iter.nextInt()]
    })
  }

//  operator fun plusAssign(d: Int) {
//    dims!! += d
//  }
  
  override fun iterator() = dims!!.iterator()
  
  override fun toString(): String {
    val sb = StringBuilder()
    sb.append("(")
    for ((i, value) in dims!!.withIndex()) {
      sb.append(if (value == -1) "?" else value)
      if (i < dims!!.lastIndex)
        sb.append(", ")
    }
    sb.append(")")
    return sb.toString()
  }
  
  fun isCompatibleWith(other: Shape): Boolean {
    if (rank() != other.rank()) return false
    for (i in 0 until rank()) {
      if (!compatible(this[i], other[i]))
        return false
    }
    return true
  }
  
  fun compatible(d1: Int, d2: Int) = when {
    d1 == d2 -> true
    d1 == -1 || d2 == -1 -> true
    else -> false
  }
}