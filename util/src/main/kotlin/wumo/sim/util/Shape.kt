@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.util

import java.util.*

val scalarDimension = Shape(IntArray(0))
//infix fun Int.x(a: Int) = Shape(this, a)
//infix fun Shape.x(a: Int) = run {
//  val dims = asIntArray()
//  if (dims == null) Shape()
//  else Shape(*dims, a)
//}
//
//infix fun Int.x(shape: Shape) = run {
//  val dims = shape.asIntArray()
//  if (dims == null) Shape()
//  else Shape(this, *dims)
//}
//
//infix fun Shape.x(shape: Shape) = run {
//  val dims1 = asIntArray()
//  val dims2 = shape.asIntArray()
//  if (dims1 == null || dims2 == null) Shape()
//  else Shape(*dims1, *dims2)
//}

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
  
  companion object {
    operator fun invoke(vararg d: Int) = Shape(d)
    operator fun invoke(d: Int, shape: Shape): Shape {
      val dims = shape.asIntArray()
      return if (dims == null) Shape(d)
      else Shape(d, *dims)
    }
    
    operator fun invoke(dims: Collection<Int>): Shape =
        Shape(dims.toIntArray())
    
    fun unknown(rank: Int = -1): Shape =
        if (rank == -1) Shape()
        else Shape(IntArray(rank) { -1 })
  }
  
  fun asLongArray() = if (dims == null) null
  else
    LongArray(dims.size) { dims[it].toLong() }
  
  fun asIntArray() = dims
  
  val rank = dims?.size ?: -1
  
  val isFullyDefined: Boolean
    get() = dims?.all { it >= 0 } ?: false
  
  val isUnknown: Boolean
    get() = !isFullyDefined
  val isScalar = dims?.size == 0
  
  fun numElements() =
      if (!isFullyDefined) -1
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
    if (dims == null) Shape()
    else {
      val size = (idx.endInclusive - idx.start) / idx.step
      val iter = idx.iterator()
      Shape(IntArray(size) {
        this[iter.nextInt()]
      })
    }
  }
  
  operator fun set(idx: Int, d: Int) {
    dims!![idx] = d
  }
  
  operator fun plus(d: Int): Shape =
      plus(Shape(d))
  
  operator fun plus(shape: Shape): Shape =
      when {
        dims == null -> shape.copy()
        shape.dims == null -> copy()
        else -> Shape(*dims, *shape.dims)
      }
  
  @Suppress("NAME_SHADOWING")
  fun slice(start: Int, end: Int? = null, step: Int = 1): Shape =
      if (dims == null) Shape()
      else {
        var end = end ?: if (start >= 0) dims.size else 0
        if (end < 0) end += dims.size
        val start = if (start < 0) start + dims.size else start
        val size = (end - start) / step
        val iter = (start until end step step).iterator()
        Shape(IntArray(size) {
          this[iter.nextInt()]
        })
      }
  
  override fun iterator() = dims!!.iterator()
  
  override fun toString(): String {
    val sb = StringBuilder()
    sb.append("(")
    if (dims != null)
      for ((i, value) in dims!!.withIndex()) {
        sb.append(if (value == -1) "?" else value)
        if (i < dims.lastIndex)
          sb.append(", ")
      }
    else
      sb.append("unknow_rank:true")
    sb.append(")")
    return sb.toString()
  }
  
  fun isCompatibleWith(other: Shape) = run {
    when {
      rank == -1 || other.rank == -1 -> true
      rank != other.rank -> false
      else -> (0 until rank).all { compatible(this[it], other[it]) }
    }
  }
  
  inline fun compatible(d1: Int, d2: Int) =
      d1 == -1 || d2 == -1 || d1 == d2
  
  fun mergeWith(other: Shape): Shape {
    return when {
      rank == -1 -> other
      other.rank == -1 -> this
      else -> {
        assertSameRank(other)
        assertIsCompatibleWith(other)
        Shape(this.dims!!.zip(other.dims!!).map { (_1, _2) ->
          when {
            _1 == -1 -> _2
            _2 == -1 -> _1
            else -> _1
          }
        }.toIntArray())
      }
    }
  }
  
  fun withRank(rank: Int): Shape = mergeWith(unknown(rank))
  
  fun withRankAtLeast(rank: Int): Shape {
    assertRankAtLeast(rank)
    return this
  }
  
  fun assertSameRank(other: Shape) =
      errorIf(this.rank != other.rank) {
        "Shape '$this' must have the same rank as shape '$other'"
      }
  
  fun assertRankAtLeast(rank: Int) =
      errorIf(this.rank < rank) {
        "Shape '$this' must have rank at least $rank."
      }
  
  fun assertIsCompatibleWith(other: Shape) =
      errorIf(!isCompatibleWith(other)) {
        "Shape '$this' must be compatible with shape '$other'."
      }
  
  fun copy(): Shape = Shape(dims?.copyOf())
  
  override fun equals(other: Any?): Boolean {
    if (this === other) return true
    if (javaClass != other?.javaClass) return false
    
    other as Shape
    
    if (!Arrays.equals(dims, other.dims)) return false
    
    return true
  }
  
  override fun hashCode(): Int {
    return dims?.let { Arrays.hashCode(it) } ?: 0
  }
  
}

fun Int.isCompatibleWith(other: Int): Boolean =
    this == -1 || other == -1 || this == other
  