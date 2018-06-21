package wumo.sim.algorithm.util

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.Scope.NewRootScope
import org.bytedeco.javacpp.tensorflow.Variable
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test

class TFHelperTest {
  @Before
  fun setup() {
    Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
    tensorflow.InitMain("trainer", null as IntArray?, null)
  }
  
  @Test
  fun `init variable`() {
    val scope = NewRootScope()
    val eps = Variable(scope.WithOpName(""))
    
  }
  
  @After
  fun tear() {
  
  }
}