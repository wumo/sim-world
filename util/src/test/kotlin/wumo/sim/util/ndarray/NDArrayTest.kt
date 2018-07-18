package wumo.sim.util.ndarray

import org.junit.Assert.*
import org.junit.Test
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
    println(c)
  }
}