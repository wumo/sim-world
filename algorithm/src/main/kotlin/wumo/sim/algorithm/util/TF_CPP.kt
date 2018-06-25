@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.algorithm.util

import org.bytedeco.javacpp.tensorflow.*

class TF_CPP(val scope: Scope) {
  val trainables = mutableListOf<_Variable>()
  val init_ops = mutableListOf<_Assign>()
  
  inline fun scope(name: String) = if (name.isEmpty()) scope else scope.WithOpName(name)
  
  fun placeholder(shape: Dimension, name: String = "") =
      _Placeholder(Placeholder(scope(name), DT_FLOAT,
          Placeholder.Shape(TensorShape(*shape.asLongArray()).asPartialTensorShape())))
  
  fun _variable(shape: Dimension, name: String = "", trainable: Boolean = true) =
      _Variable(Variable(scope.WithOpName(name),
          TensorShape(*shape.asLongArray()).asPartialTensorShape(), DT_FLOAT), name)
          .apply {
            if (trainable)
              trainables += this
          }
  
  fun variable(shape: Dimension, name: String = "", trainable: Boolean = true) = variable(shape, 0f, name, trainable)
  
  fun variable(shape: Dimension, initial_value: Float, name: String = "", trainable: Boolean = true) =
      _Variable(Variable(scope(name),
          TensorShape(*shape.asLongArray()).asPartialTensorShape(), DT_FLOAT), name)
          .apply {
            val assign = "$name/assign"
            val assignOp = _Assign(Assign(scope.WithOpName(assign), this.asInput(),
                Input(Const(scope, initial_value, TensorShape(*shape.asLongArray())))), assign)
            if (trainable)
              trainables += this
            init_ops += assignOp
          }
  
  fun variable(shape: Dimension, initializer: Node, name: String = "", trainable: Boolean = true) =
      _Variable(Variable(scope(name),
          TensorShape(*shape.asLongArray()).asPartialTensorShape(), DT_FLOAT), name)
          .apply {
            val assign = "$name/assign"
            val assignOp = _Assign(Assign(scope.WithOpName(assign), this.asInput(), initializer.asInput()), assign)
            if (trainable)
              trainables += this
            init_ops += assignOp
          }
  
  fun assign(a: Node, b: Node, name: String = "") = Assign(scope(name), a.asInput(), b.asInput())
  
  fun matmul(a: Node, b: Node, name: String = "") = _MatMul(MatMul(scope(name), a.asInput(), b.asInput()))
  
  fun argmax(a: Node, dim: Node, name: String = "") = _ArgMax(ArgMax(scope(name), a.asInput(), dim.asInput()))
  
  fun sum(a: Node, axis: Node, name: String = "") = _Sum(Sum(scope(name), a.asInput(), axis.asInput()))
  
  fun square(a: Node, name: String = "") = _Square(Square(scope(name), a.asInput()))
  
  fun subtract(a: Node, b: Node, name: String = "") = _Sub(Subtract(scope(name), a.asInput(), b.asInput()))
  
  fun tensor(vararg data: Int) = _Tensor(Tensor.create(data, TensorShape(*longArrayOf(data.size.toLong()))))
  fun tensor(vararg data: Long) = _Tensor(Tensor.create(data, TensorShape(*longArrayOf(data.size.toLong()))))
  fun tensor(vararg data: Int, shape: Dimension) = _Tensor(Tensor.create(data, TensorShape(*shape.asLongArray())))
  
  fun GradientDescentOptimizer(learningRate: Float, loss: Node, name: String = "") {
    val node_outputs = OutputVector(loss.asOutput())
    val param_outputs = Array(trainables.size) { trainables[it].asOutput() }
    val node_inputs = OutputVector(*param_outputs)
    val node_grad_outputs = OutputVector()
    TF_CHECK_OK(AddSymbolicGradients(scope, node_outputs, node_inputs, node_grad_outputs))
    val alpha = const(learningRate)
    for ((i, trainable) in trainables.withIndex())
      ApplyGradientDescent(scope("${trainable.name}/$name"), trainable.asInput(), alpha.asInput(), Input(node_grad_outputs[i.toLong()]))
  }
  
  fun random_uniform(shape: Dimension): _RandomUniform {
    return _RandomUniform(RandomUniform(scope, tensor(*shape.asLongArray()).asInput(), DT_FLOAT))
  }
  
  fun random_uniform(shape: Dimension, min: Float, max: Float): _Add {
//    val max = Const(scope, max)
//    val min = Const(scope, min)
//    val rand = RandomUniform(scope, Input(Tensor.create(shape.asLongArray(), TensorShape(*longArrayOf(shape.rank())))), DT_FLOAT)
//    return _Add(Add(scope, Multiply(scope, rand.asInput(), Subtract(scope, Input(max), Input(min)).asInput()).asInput(), Input(min)))
    val rand = random_uniform(shape)
    val max = const(max)
    val min = const(min)
    return rand * (max - min) + min
  }
  
  fun const(data: Boolean) = _Const(Const(scope, data))
  fun const(data: Byte) = _Const(Const(scope, data))
  fun const(data: Short) = _Const(Const(scope, data))
  fun const(data: Int) = _Const(Const(scope, data))
  fun const(data: Float) = _Const(Const(scope, data))
  fun const(data: Long) = _Const(Const(scope, data))
  fun const(data: Double) = _Const(Const(scope, data))
  
  operator fun Node.plus(a: Node) = _Add(Add(scope, this.asInput(), a.asInput()))
  operator fun Node.times(a: Node) = _Mul(Multiply(scope, this.asInput(), a.asInput()))
  operator fun Node.minus(a: Node) = _Sub(Subtract(scope, this.asInput(), a.asInput()))
  
  fun session(block: Session.() -> Unit) {
    val def = GraphDef()
    TF_CHECK_OK(scope.ToGraphDef(def))
    val session = Session(SessionOptions())
    TF_CHECK_OK(session.Create(def))
    block(session)
    session.close()
    def.close()
  }
}