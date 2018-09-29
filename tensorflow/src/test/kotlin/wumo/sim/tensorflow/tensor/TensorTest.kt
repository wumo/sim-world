package wumo.sim.tensorflow.tensor

import org.junit.Assert.*
import org.junit.Test
import wumo.sim.tensorflow.ops.BaseTest
import wumo.sim.tensorflow.tf
import wumo.sim.util.Shape
import wumo.sim.util.ndarray.NDArray
import wumo.sim.util.ndarray.types.NDFloat

class TensorTest :BaseTest(){
  @Test
  fun testFromNDArray() {
    tf
    val a = NDArray(Shape(1, 16), NDFloat) { if (it == 0) 1f else 0f }
    val b=Tensor.fromNDArray(a)
    
    println(a)
    println(b)
  }
}