package wumo.sim.util.ndarray

import org.apache.commons.math3.linear.MatrixUtils
import org.junit.Test
import wumo.sim.util.a
import wumo.sim.util.d

class LinalgKtTest {
  
  @Test
  fun svd() {
    val A = MatrixUtils.createRealMatrix(a(
        d(1.0, 1.0, 1.0, 0.0, 0.0),
        d(3.0, 3.0, 3.0, 0.0, 0.0),
        d(4.0, 4.0, 4.0, 0.0, 0.0),
        d(5.0, 5.0, 5.0, 0.0, 0.0),
        d(0.0, 2.0, 0.0, 4.0, 4.0),
        d(0.0, 0.0, 0.0, 5.0, 5.0),
        d(0.0, 1.0, 0.0, 2.0, 2.0)
    ))
    println(A.toNDArray())
    val (u, s, vt) = svd(A.toNDArray())
    println("U:\n$u")
    println("U:\n$s")
    println("U:\n$vt")
  }
}