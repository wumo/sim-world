package wumo.sim.algorithm.util.c_api

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.TF_Version
import org.junit.Before
import org.junit.Test

class TF_CTest {
  @Before
  fun setup() {
    Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
    tensorflow.InitMain("trainer", null as IntArray?, null)
  }
  
  @Test
  fun `print version`() {
    println(TF_Version().string)
  }
  
  @Test
  fun `const test`() {
  
  }
}