package wumo.sim.util.ndarray

import org.junit.Assert
import org.junit.Test
import wumo.sim.util.Shape
import wumo.sim.util.l

class NDArrayTest {
  @Test
  fun toNDArray() {
    val a = NDArray.toNDArray(1)
    println(a)
    val d = NDArray.toNDArray(Integer(1))
    println(d)
    val b = NDArray.toNDArray(2f)
    println(b)
    val c = NDArray.toNDArray(l(1L, 2L, 3L))
    val e = NDArray.toNDArray(arrayOf("h", "e", "l", "l", "o"))
    println(e)
    println(c)
    val f = NDArray.toNDArray(arrayOf(1L, 2L, 3L))
    println(f)
    val g = NDArray.toNDArray(f)
    Assert.assertEquals(f, g)
    val h = NDArray.toNDArray(listOf(1L, 2L, 3L))
    println(h)
    val i = NDArray.toNDArray(arrayOf(NDArray(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f)),
                                      NDArray(Shape(2, 2), floatArrayOf(5f, 6f, 7f, 8f)),
                                      NDArray(Shape(2, 2), floatArrayOf(9f, 10f, 11f, 12f))))
    println(i)
    val j = NDArray.toNDArray(listOf(NDArray(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f)),
                                     NDArray(Shape(2, 2), floatArrayOf(5f, 6f, 7f, 8f)),
                                     NDArray(Shape(2, 2), floatArrayOf(9f, 10f, 11f, 12f))))
    println(j)
    
    val k = NDArray.toNDArray(arrayOf(arrayOf(NDArray(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f)),
                                              NDArray(Shape(2, 2), floatArrayOf(5f, 6f, 7f, 8f)),
                                              NDArray(Shape(2, 2), floatArrayOf(9f, 10f, 11f, 12f))),
                                      arrayOf(NDArray(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f)),
                                              NDArray(Shape(2, 2), floatArrayOf(5f, 6f, 7f, 8f)),
                                              NDArray(Shape(2, 2), floatArrayOf(9f, 10f, 11f, 12f))),
                                      arrayOf(NDArray(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f)),
                                              NDArray(Shape(2, 2), floatArrayOf(5f, 6f, 7f, 8f)),
                                              NDArray(Shape(2, 2), floatArrayOf(9f, 10f, 11f, 12f)))))
    println(k)
    
    val L = NDArray.toNDArray(listOf(arrayOf(NDArray(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f)),
                                             NDArray(Shape(2, 2), floatArrayOf(5f, 6f, 7f, 8f)),
                                             NDArray(Shape(2, 2), floatArrayOf(9f, 10f, 11f, 12f))),
                                     arrayOf(NDArray(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f)),
                                             NDArray(Shape(2, 2), floatArrayOf(5f, 6f, 7f, 8f)),
                                             NDArray(Shape(2, 2), floatArrayOf(9f, 10f, 11f, 12f))),
                                     arrayOf(NDArray(Shape(2, 2), floatArrayOf(1f, 2f, 3f, 4f)),
                                             NDArray(Shape(2, 2), floatArrayOf(5f, 6f, 7f, 8f)),
                                             NDArray(Shape(2, 2), floatArrayOf(9f, 10f, 11f, 12f)))))
    println(L)
  }
}