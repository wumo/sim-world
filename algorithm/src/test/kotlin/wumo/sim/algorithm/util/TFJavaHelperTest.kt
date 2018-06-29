package wumo.sim.algorithm.util

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import org.bytedeco.javacpp.tensorflow.Scope.NewRootScope
import org.junit.After
import org.junit.Before
import org.junit.Test
import wumo.sim.algorithm.util.cpp_api.TF_CPP
import java.nio.FloatBuffer

class TFJavaHelperTest {
  @Before
  fun setup() {
    Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
    tensorflow.InitMain("trainer", null as IntArray?, null)
  }
  
  @Test
  fun `init variable`() {
    val scope = NewRootScope()
    val eps = Variable(scope.WithOpName("W"), TensorShape(2, 3).asPartialTensorShape(), DT_FLOAT)
    val tensorProto = TensorProto()
    tensorProto.set_dtype(DT_FLOAT)
    tensorProto.mutable_tensor_shape().apply {
      add_dim().set_size(2)
      add_dim().set_size(3)
    }
    tensorProto.add_float_val(9f)
    val const = ConstFromProto(scope.WithOpName("init_value"), tensorProto)
    val assign = Assign(scope.WithOpName("assign"), eps.asInput(), Input(const))
    val assign2 = Assign(scope.WithOpName("assign2"), eps.asInput(), Input(const))
    val init = NoOp(scope.WithOpName("init").WithControlDependencies(assign.asOutput())
        .WithControlDependencies(assign2.asOutput()))
    
    val def = GraphDef()
    TF_CHECK_OK(scope.ToGraphDef(def))
    TF_CHECK_OK(tensorflow.WriteTextProto(Env.Default(), "resources/controlInput.pbtxt", def))
  }
  
  @Test
  fun `variable test`() {
//    val tf = TF_CPP(NewRootScope())
//    val A = tf.variable(2 x 3, 9f, "A")
//    println(tf.debugString())
  }
  
  @Test
  fun `init variable 2`() {
    val scope = NewRootScope()
    val subscope = scope.WithOpName("W")
    val eps = Variable(subscope.WithOpName("W"), TensorShape(2, 3).asPartialTensorShape(), DT_FLOAT)
    val const = Const(subscope.NewSubScope("W"), 9f, TensorShape(2, 3))
    val assign = Assign(subscope.NewSubScope("W"), eps.asInput(), Input(const))
    val def = GraphDef()
    TF_CHECK_OK(scope.ToGraphDef(def))
//    TF_CHECK_OK(tensorflow.WriteTextProto(Env.Default(), "resources/custom2.pbtxt", def))
    println(DebugStringWhole(def).string)
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
  
  @Test
  fun `test helpers`() {
    val scope = NewRootScope()
    val tf = TF_CPP(scope)
    val W = Variable(scope.WithOpName("W"), TensorShape(16, 4).asPartialTensorShape(), DT_FLOAT)
    val assign_W = Assign(scope.WithOpName("assign_W"),
        W.asInput(),
        Div(scope,
            RandomUniform(scope,
                Input(Tensor.create(intArrayOf(16, 4), TensorShape(*longArrayOf(2)))), DT_FLOAT).asInput(),
            Input(Const(scope, 10f))).asInput())
    
    tf.session {
      val outputs = TensorVector()
//      Run(StringTensorPairVector(), StringVector(), StringVector("assign_W:0"), outputs)
//      Run(StringTensorPairVector(), StringVector("W:0"), StringVector(), outputs)
      val w = outputs[0].createBuffer<FloatBuffer>()
      while (w.hasRemaining()) {
        println(w.get())
      }
    }
    
  }
  
  @Test
  fun `visit graph node`() {
    val def = GraphDef()
//  val loc = ResourceLoader.getResource("def.pb")
    TF_CHECK_OK(tensorflow.ReadTextProto(Env.Default(), "resources/train.pbtxt", def))
    val node_count = def.node_size()
    println("$node_count nodes in graph")
    for (i in 0 until node_count) {
      val n = def.node(i)
      
      println(n.name().string)
    }
    def.close()
  }
  
  @After
  fun tear() {
  
  }
}