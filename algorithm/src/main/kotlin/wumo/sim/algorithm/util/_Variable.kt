package wumo.sim.algorithm.util

import org.bytedeco.javacpp.tensorflow.*

interface Node {
  fun asInput(): Input
  fun asOutput(): Output
}

class _Tensor(val ref: Tensor) : Node {
  override fun asOutput() = Output(ref)
  
  override fun asInput() = Input(ref)
}

class _Variable(val ref: Variable, val name: String) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}

class _Placeholder(val ref: Placeholder) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}

class _Const(val ref: Output) : Node {
  override fun asInput() = Input(ref)
  override fun asOutput() = Output(ref)
}

class _RandomUniform(val ref: RandomUniform) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}

class _Add(val ref: Add) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}

class _Mul(val ref: Multiply) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}

class _Sub(val ref: Subtract) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}

class _Div(val ref: Div) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}

class _MatMul(val ref: MatMul) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}

class _ArgMax(val ref: ArgMax) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}

class _Sum(val ref: Sum) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}

class _Square(val ref: Square) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}

class _ApplyGradientDescent(val ref: ApplyGradientDescent) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}

class _Assign(val ref: Assign, val name: String) : Node {
  override fun asInput() = ref.asInput()
  override fun asOutput() = ref.asOutput()
}