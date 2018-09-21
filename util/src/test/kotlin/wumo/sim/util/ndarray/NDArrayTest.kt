package wumo.sim.util.ndarray

import org.junit.Assert
import org.junit.Assert.*
import org.junit.Test
import wumo.sim.util.Shape
import wumo.sim.util.l
import java.util.*

class NDArrayTest {
  @Test
  fun toNDArray() {
    val a = NDArray.toNDArray(1)
    assertEquals(1, a.size)
    assertEquals(Integer::class.java, a.dtype)
    assertEquals(1, a.get())
    
    val d = NDArray.toNDArray(Integer(1))
    assertEquals(1, d.size)
    assertEquals(Integer::class.java, d.dtype)
    assertEquals(1, d.get())
    
    val b = NDArray.toNDArray(2f)
    assertEquals(1, b.size)
    assertEquals(java.lang.Float::class.java, b.dtype)
    assertEquals(2f, b.get())
    
    val c = NDArray.toNDArray(l(1L, 2L, 3L))
    assertEquals(3, c.size)
    assertEquals(java.lang.Long::class.java, c.dtype)
    
    val e = NDArray.toNDArray(arrayOf("h", "e", "l", "l", "o"))
    println(e)
    println(c)
    val f = NDArray.toNDArray(arrayOf(1L, 2L, 3L))
    println(f)
    val g = NDArray.toNDArray(f)
    assertEquals(f, g)
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
  
  @Test
  fun slice() {
    val m = NDArray(Shape(2, 2, 2),
                    floatArrayOf(1f, 2f, 3f, 4f,
                                 5f, 6f, 7f, 8f))
    println(m)
    println(m(0))
    println(m(1))
    println(m(0, 0))
    println(m(0, 1))
    println(m(1, 0))
    println(m(1, 1))
    println(m(0, 0, 0))
    println(m(0, 0, 1))
  }
  
  inline fun <reified T : Any> make(shape: Shape, initvalue: T): NDArray<T> =
      NDArray(shape, initvalue)
  
  @Test
  fun shapeNDArray() {
    
    val n1 = make(Shape(2, 2), 1)
    println(n1)
  }
  
  @Test
  fun setTest() {
    val a = NDArray(Shape(2, 3)) { it }
    assertEquals(Shape(2, 3), a.shape)
    val b = NDArray(Shape(2, 3)) { it + 6 }
    val c = NDArray(Shape(2, 2, 3), 0)
    c[0] = a
    c[1] = b
    var expected = 0
    for (i in c) {
      assertEquals(expected++, i)
    }
    println(a)
    println(b)
    println(c)
  }
  
  @Test
  fun initTest() {
    val idx = arrayOf(
        intArrayOf(0, 0),
        intArrayOf(0, 1),
        intArrayOf(0, 2),
        intArrayOf(1, 0),
        intArrayOf(1, 1),
        intArrayOf(1, 2)
    )
    var i = 0
    val a = NDArray.from(Shape(2, 3)) {
      Arrays.toString(it).apply {
        assertArrayEquals(idx[i], it)
        i++
      }
    }
    println(a)
  }
  
  @Test
  fun reduce() {
    val a = NDArray(Shape(2, 3)) { it }
    val b = a.reduce(0) { acc, i -> acc + i }
    println(a)
    assertEquals(3, b[0])
    assertEquals(5, b[1])
    assertEquals(7, b[2])
  }
  
  @Test
  fun max() {
    val a = NDArray(Shape(2, 3)) { it }
    val b = a.max(0)
    println(a)
    println(b)
    assertEquals(3, b[0])
    assertEquals(4, b[1])
    assertEquals(5, b[2])
  }
  
  @Test
  fun min() {
    val a = NDArray(Shape(2, 3)) { it }
    val b = a.min(0)
    println(a)
    println(b)
    assertEquals(0, b[0])
    assertEquals(1, b[1])
    assertEquals(2, b[2])
  }
}