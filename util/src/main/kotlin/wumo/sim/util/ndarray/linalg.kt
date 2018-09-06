package wumo.sim.util.ndarray

import org.apache.commons.math3.linear.MatrixUtils
import org.apache.commons.math3.linear.RealMatrix
import org.apache.commons.math3.linear.SingularValueDecomposition
import wumo.sim.util.Shape
import wumo.sim.util.t3

fun NDArray<Float>.toMatrix(): RealMatrix {
  assert(numDims == 2)
  val m = shape[0]
  val n = shape[1]
  return MatrixUtils.createRealMatrix(Array(m) { row ->
    DoubleArray(n) { col ->
      this[row, col].toDouble()
    }
  })
}

fun RealMatrix.toNDArray(): NDArray<Float> {
  val shape = Shape(rowDimension, columnDimension)
  return NDArray(shape, FloatArray(shape.numElements()) {
    val i = it / columnDimension
    val j = it % columnDimension
    getEntry(i, j).toFloat()
  })
}

/**
 * Reduced SVD
 */
fun svd(a: NDArray<Float>): t3<NDArray<Float>, NDArray<Float>, NDArray<Float>> {
  val m = a.toMatrix()
  val svd = SingularValueDecomposition(m)
  return t3(svd.u.toNDArray(), svd.s.toNDArray(), svd.vt.toNDArray())
}