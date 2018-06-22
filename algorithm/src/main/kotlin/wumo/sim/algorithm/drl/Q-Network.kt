package wumo.sim.algorithm.drl

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*
import org.bytedeco.javacpp.tensorflow.Placeholder.Shape
import org.bytedeco.javacpp.tensorflow.Scope.NewRootScope
import wumo.sim.algorithm.util.TFHelper
import wumo.sim.algorithm.util.x
import wumo.sim.envs.toy_text.FrozenLake
import wumo.sim.util.math.Rand
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.file.Files
import java.nio.file.Paths

fun main(args: Array<String>) {
  // Load all javacpp-preset classes and native libraries
  Loader.load(org.bytedeco.javacpp.tensorflow::class.java)
  
  // Platform-specific initialization routine
  tensorflow.InitMain("trainer", null as IntArray?, null)
  
  val env = FrozenLake()
  
  val scope = NewRootScope()
  val tf = TFHelper(scope)
  val inputs = tf.placeholder(1 x 16, name = "inputs1")
//  val inputs = Placeholder(scope.WithOpName("inputs1"), DT_FLOAT, Shape(TensorShape(1, 16).asPartialTensorShape()))
  val W = tf.variable(16 x 4, name = "W")
//  val W = Variable(scope.WithOpName("W"), TensorShape(16, 4).asPartialTensorShape(), DT_FLOAT)
//  val assign_W = Assign(scope.WithOpName("assign_W"),
//      W.asInput(),
//      Div(scope, RandomUniform(scope,
//          Input(Tensor.create(longArrayOf(16, 4),
//              TensorShape(*longArrayOf(2)))), DT_FLOAT).asInput(), Input(Const(scope, 10f))).asInput())
  tf.assign(W, tf.random_uniform(16 x 4, 0f, 0.1f), name = "assign_W")
//  val assign_W = Assign(scope.WithOpName("assign_W"),
//      W.asInput(),
//      Div(scope, tf.random_uniform(16 x 4).asInput(), tf.const(10f).asInput()).asInput())
  val Qout = tf.matmul(inputs, W, name = "Qout")
//  val Qout = MatMul(scope.WithOpName("Qout"), inputs.asInput(), W.asInput())
  val predict = tf.argmax(Qout, tf.const(1), name = "predict")
//  val predict = ArgMax(scope.WithOpName("predict"), Qout.asInput(), Input(tf.const(1)))
  
  val nextQ = tf.placeholder(1 x 4, name = "nextQ")
//  val nextQ = Placeholder(scope.WithOpName("nextQ"), DT_FLOAT, Shape(TensorShape(1, 4).asPartialTensorShape()))
  
  val loss = tf.sum(tf.square(tf.subtract(nextQ, Qout)), tf.tensor(0, 1))
//  val loss = Sum(scope.WithOpName("loss"), Square(scope, Subtract(scope, nextQ.asInput(), Qout.asInput()).asInput()).asInput(),
//      Input(Tensor.create(intArrayOf(0, 1), TensorShape(*longArrayOf(2)))))
  
  tf.GradientDescentOptimizer(0.1f, loss, "apply")

  val initializer=tf.global_variables_initializer()
//  val node_outputs = OutputVector(loss.asOutput())
//  val node_inputs = OutputVector(W.asOutput())
//  val node_grad_outputs = OutputVector()
//  TF_CHECK_OK(AddSymbolicGradients(scope, node_outputs, node_inputs, node_grad_outputs))
//
//  val alpha = Input(Const(scope.WithOpName("alpha"), 0.1f))
//  val apply_W = ApplyGradientDescent(scope.WithOpName("apply_W"), W.asInput(), alpha, Input(node_grad_outputs[0]))
  
  tf.session {
    val outputs = TensorVector()
    Run(StringTensorPairVector(), StringVector("W"), StringVector("assign_W"), outputs)
    
    val y = .99
    var e = 0.1
    val num_episodes = 2000
    var sum = 0.0
    for (i in 0 until num_episodes) {
      var s = env.reset()
      var rAll = 0.0
      var j = 0
      while (j < 99) {
        j++
        Run(StringTensorPairVector(arrayOf("inputs1"),
            arrayOf(Tensor.create(FloatArray(16) { if (it == s) 1f else 0f }, TensorShape(1, 16)))),
            StringVector("predict", "Qout"),
            StringVector(),
            outputs)
        val a = outputs[0]
        val allQ = FloatArray(4)
        outputs[1].createBuffer<FloatBuffer>().get(allQ)
        var _a = a.createBuffer<LongBuffer>()[0].toInt()
        if (Rand().nextDouble() < e)
          _a = env.action_space.sample()
        val (s1, r, d) = env.step(_a)
        outputs.clear()
        Run(StringTensorPairVector(arrayOf("inputs1"),
            arrayOf(Tensor.create(FloatArray(16) { if (it == s1) 1f else 0f }, TensorShape(1, 16)))),
            StringVector("Qout"),
            StringVector(),
            outputs)
        val Q1 = FloatArray(4)
        outputs[0].createBuffer<FloatBuffer>().get(Q1)
        
        val maxQ1 = Q1.max()!!
        val targetQ = allQ
        targetQ[_a] = (r + y * maxQ1).toFloat()
        
        outputs.clear()
        Run(StringTensorPairVector(arrayOf("inputs1", "nextQ"),
            arrayOf(Tensor.create(FloatArray(16) { if (it == s) 1f else 0f }, TensorShape(1, 16)),
                Tensor.create(targetQ, TensorShape(1, 4)))),
            StringVector(),
            StringVector("apply_W"),
            outputs)
        rAll += r
        s = s1
        if (d)
          e = 1.0 / (i / 50 + 10)
      }
      println("$rAll-$i")
      sum += rAll
    }
    println(sum / num_episodes)
  }

//  val def = GraphDef()
//  TF_CHECK_OK(scope.ToGraphDef(def))
//  TF_CHECK_OK(tensorflow.WriteTextProto(Env.Default(), "resources/custom.pbtxt", def))
//  val session = Session(SessionOptions())
////  SetDefaultDevice("/gpu:0", def)
//  TF_CHECK_OK(session.Create(def))
//  val outputs = TensorVector()
//  session.Run(StringTensorPairVector(), StringVector("W"), StringVector("assign_W"), outputs)
//
//  val y = .99
//  var e = 0.1
//  val num_episodes = 2000
//  var sum = 0.0
//  for (i in 0 until num_episodes) {
//    var s = env.reset()
//    var rAll = 0.0
//    var j = 0
//    while (j < 99) {
//      j++
//      session.Run(StringTensorPairVector(arrayOf("inputs1"),
//          arrayOf(Tensor.create(FloatArray(16) { if (it == s) 1f else 0f }, TensorShape(1, 16)))),
//          StringVector("predict", "Qout"),
//          StringVector(),
//          outputs)
//      val a = outputs[0]
//      val allQ = FloatArray(4)
//      outputs[1].createBuffer<FloatBuffer>().get(allQ)
//      var _a = a.createBuffer<LongBuffer>()[0].toInt()
//      if (Rand().nextDouble() < e)
//        _a = env.action_space.sample()
//      val (s1, r, d) = env.step(_a)
//      outputs.clear()
//      session.Run(StringTensorPairVector(arrayOf("inputs1"),
//          arrayOf(Tensor.create(FloatArray(16) { if (it == s1) 1f else 0f }, TensorShape(1, 16)))),
//          StringVector("Qout"),
//          StringVector(),
//          outputs)
//      val Q1 = FloatArray(4)
//      outputs[0].createBuffer<FloatBuffer>().get(Q1)
//
//      val maxQ1 = Q1.max()!!
//      val targetQ = allQ
//      targetQ[_a] = (r + y * maxQ1).toFloat()
//
//      outputs.clear()
//      session.Run(StringTensorPairVector(arrayOf("inputs1", "nextQ"),
//          arrayOf(Tensor.create(FloatArray(16) { if (it == s) 1f else 0f }, TensorShape(1, 16)),
//              Tensor.create(targetQ, TensorShape(1, 4)))),
//          StringVector(),
//          StringVector("apply_W"),
//          outputs)
//      rAll += r
//      s = s1
//      if (d)
//        e = 1.0 / (i / 50 + 10)
//    }
//    println("$rAll-$i")
//    sum += rAll
//  }
//  println(sum / num_episodes)
//  session.close()
//  def.close()
}
