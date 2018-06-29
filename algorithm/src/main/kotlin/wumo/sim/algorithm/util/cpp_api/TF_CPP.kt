@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.algorithm.util.cpp_api

import org.bytedeco.javacpp.Loader
import org.bytedeco.javacpp.tensorflow
import org.bytedeco.javacpp.tensorflow.*

class TF_CPP(val root: Scope = Scope.NewRootScope()) {
  companion object {
    init {
      Loader.load(tensorflow::class.java)
      tensorflow.InitMain("trainer", null as IntArray?, null)
    }
  }
  
  //  val trainables = mutableListOf<_Variable>()
  val init_ops = mutableListOf<Output>()
  
  //  fun placeholder(shape: Dimension, name: String = "") =
//      _Placeholder(Placeholder(root(name), DT_FLOAT,
//          Placeholder.Shape(TensorShape(*shape.asLongArray()).asPartialTensorShape())))
//
  fun debugString(): String {
    GraphDef().use { def ->
      TF_CHECK_OK(root.ToGraphDef(def))
      return DebugStringWhole(def).string
    }
  }
  
  fun writeTextProto(fileName: String) {
    GraphDef().use { def ->
      TF_CHECK_OK(root.ToGraphDef(def))
      TF_CHECK_OK(tensorflow.WriteTextProto(Env.Default(), fileName, def))
    }
  }
//
//  fun assign(a: Node, b: Node, name: String = "") = Assign(root(name), a.asInput(), b.asInput())
//
//  fun matmul(a: Node, b: Node, name: String = "") = _MatMul(MatMul(root(name), a.asInput(), b.asInput()))
//
//  fun argmax(a: Node, dim: Node, name: String = "") = _ArgMax(ArgMax(root(name), a.asInput(), dim.asInput()))
//
//  fun sum(a: Node, axis: Node, name: String = "") = _Sum(Sum(root(name), a.asInput(), axis.asInput()))
//
//  fun square(a: Node, name: String = "") = _Square(Square(root(name), a.asInput()))
//
//  fun subtract(a: Node, b: Node, name: String = "") = _Sub(Subtract(root(name), a.asInput(), b.asInput()))
//
//  fun tensor(vararg data: Int) = _Tensor(Tensor.tensor(data, TensorShape(*longArrayOf(data.size.toLong()))))
//  fun tensor(vararg data: Long) = _Tensor(Tensor.tensor(data, TensorShape(*longArrayOf(data.size.toLong()))))
//  fun tensor(vararg data: Int, shape: Dimension) = _Tensor(Tensor.tensor(data, TensorShape(*shape.asLongArray())))
//
//  fun GradientDescentOptimizer(learningRate: Float, loss: Node, name: String = "") {
//    val node_outputs = OutputVector(loss.asOutput())
//    val param_outputs = Array(trainables.size) { trainables[it].asOutput() }
//    val node_inputs = OutputVector(*param_outputs)
//    val node_grad_outputs = OutputVector()
//    TF_CHECK_OK(AddSymbolicGradients(root, node_outputs, node_inputs, node_grad_outputs))
//    val alpha = const(learningRate)
//    for ((i, trainable) in trainables.withIndex())
//      ApplyGradientDescent(root("${trainable.name}/$name"), trainable.asInput(), alpha.asInput(), Input(node_grad_outputs[i.toLong()]))
//  }
//
//  fun random_uniform(shape: Dimension): _RandomUniform {
//    return _RandomUniform(RandomUniform(root, tensor(*shape.asLongArray()).asInput(), DT_FLOAT))
//  }
//
//  fun random_uniform(shape: Dimension, min: Float, max: Float): _Add {
////    val max = Const(root, max)
////    val min = Const(root, min)
////    val rand = RandomUniform(root, Input(Tensor.tensor(shape.asLongArray(), TensorShape(*longArrayOf(shape.rank())))), DT_FLOAT)
////    return _Add(Add(root, Multiply(root, rand.asInput(), Subtract(root, Input(max), Input(min)).asInput()).asInput(), Input(min)))
//    val rand = random_uniform(shape)
//    val max = const(max)
//    val min = const(min)
//    return rand * (max - min) + min
//  }
//
//  fun const(data: Boolean) = _Const(Const(root, data))
//  fun const(data: Byte) = _Const(Const(root, data))
//  fun const(data: Short) = _Const(Const(root, data))
//  fun const(data: Int) = _Const(Const(root, data))
//  fun const(data: Float) = _Const(Const(root, data))
//  fun const(data: Long) = _Const(Const(root, data))
//  fun const(data: Double) = _Const(Const(root, data))
//
//  operator fun Node.plus(a: Node) = _Add(Add(root, this.asInput(), a.asInput()))
//  operator fun Node.times(a: Node) = _Mul(Multiply(root, this.asInput(), a.asInput()))
//  operator fun Node.minus(a: Node) = _Sub(Subtract(root, this.asInput(), a.asInput()))
  
  fun session(block: SessionHelper.() -> Unit) {
    val def = GraphDef()
    TF_CHECK_OK(root.ToGraphDef(def))
    val session = Session(SessionOptions())
    TF_CHECK_OK(session.Create(def))
    block(SessionHelper(session))
    session.close()
    def.close()
  }
  
  fun global_variable_initializer(): Operation {
    var scope = root.WithOpName("init")
    for (init_op in init_ops) {
      scope = scope.WithControlDependencies(init_op)
    }
    return NoOp(scope).asOperation()
  }
}