package wumo.sim.algorithm.util.cpp_api.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.cpp_api.TF_CPP

fun TF_CPP.add(a: Output, b: Output, name: String = "") =
    Add(scope.WithOpName(name), Input(a), Input(b)).asOutput()

fun TF_CPP.sub(a: Output, b: Output, name: String = "") =
    Subtract(scope.WithOpName(name), Input(a), Input(b)).asOutput()

fun TF_CPP.mul(a: Output, b: Output, name: String = "") =
    Multiply(scope.WithOpName(name), Input(a), Input(b)).asOutput()

fun TF_CPP.div(a: Output, b: Output, name: String = "") =
    Div(scope.WithOpName(name), Input(a), Input(b)).asOutput()

fun TF_CPP.sum(a: Output, axis: Output, name: String = "") =
    Sum(scope.WithOpName(name), Input(a), Input(axis)).asOutput()

fun TF_CPP.matmul(a: Output, b: Output, name: String = "") =
    MatMul(scope.WithOpName(name), Input(a), Input(b)).asOutput()

fun TF_CPP.argmax(a: Output, dim: Int, name: String = "") =
    ArgMax(scope.WithOpName(name), Input(a), Input(const(dim, name = "dimension"))).asOutput()

fun TF_CPP.argmax(a: Output, dim: Output, name: String = "") =
    ArgMax(scope.WithOpName(name), Input(a), Input(dim)).asOutput()

fun TF_CPP.square(a: Output, name: String = "") =
    Square(scope.WithOpName(name), Input(a)).asOutput()
