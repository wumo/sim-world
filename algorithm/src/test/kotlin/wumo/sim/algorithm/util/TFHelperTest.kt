package wumo.sim.algorithm.util

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import org.bytedeco.javacpp.tensorflow.Scope.NewRootScope
import org.junit.After
import org.junit.Before
import org.junit.Test
import java.nio.FloatBuffer

class TFHelperTest {
  @Before
  fun setup() {
    Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
    tensorflow.InitMain("trainer", null as IntArray?, null)
  }
  
  @Test
  fun `init variable`() {
    val scope = NewRootScope()
    val eps = Variable(scope.WithOpName("W"), TensorShape(2, 3).asPartialTensorShape(), DT_FLOAT)
    val const = Const(scope.WithOpName("init_value"), 9f, TensorShape(2, 3))
    val assign = Assign(scope.WithOpName("assign"), eps.asInput(), Input(const))
    val def = GraphDef()
    TF_CHECK_OK(scope.ToGraphDef(def))
    TF_CHECK_OK(tensorflow.WriteTextProto(Env.Default(), "resources/custom2.pbtxt", def))
    val session = Session(SessionOptions())
    TF_CHECK_OK(session.Create(def))
    val outputs = TensorVector()
    session.Run(StringTensorPairVector(), StringVector(), StringVector("assign"), outputs)
    session.Run(StringTensorPairVector(), StringVector("W"), StringVector(), outputs)
    for (i in 0 until outputs.size()) {
      val result = outputs[i]
      val buf = result.createBuffer<FloatBuffer>()
      while (buf.hasRemaining())
        println(buf.get())
    }
    session.close()
    def.close()
  }
  
  @After
  fun tear() {
  
  }
}