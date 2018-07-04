package wumo.sim.algorithm.util.cpp_api.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.cpp_api.TF_CPP

fun TF_CPP.add(a: Output, b: Output, name: String = "", scope: Scope = root) =
    Add(scope.WithOpName(name), Input(a), Input(b)).asOutput()

fun TF_CPP.sub(a: Output, b: Output, name: String = "", scope: Scope = root) =
    Subtract(scope.WithOpName(name), Input(a), Input(b)).asOutput()

fun TF_CPP.mul(a: Output, b: Output, name: String = "", scope: Scope = root) =
    Multiply(scope.WithOpName(name), Input(a), Input(b)).asOutput()

fun TF_CPP.div(a: Output, b: Output, name: String = "", scope: Scope = root) =
    Div(scope.WithOpName(name), Input(a), Input(b)).asOutput()

fun TF_CPP.sum(a: Output, axis: Output, name: String = "", scope: Scope = root) =
    Sum(scope.WithOpName(name), Input(a), Input(axis)).asOutput()

fun TF_CPP.matmul(a: Output, b: Output, name: String = "", scope: Scope = root) =
    MatMul(scope.WithOpName(name), Input(a), Input(b)).asOutput()

fun TF_CPP.argmax(a: Output, dim: Int, name: String = "", scope: Scope = root) =
    scope.NewSubScope(name).let { s ->
      argmax(a, const(dim, "dimension", s), scope = s)
    }

fun TF_CPP.argmax(a: Output, dim: Output, name: String = "", scope: Scope = root) =
    ArgMax(scope.WithOpName(name), Input(a), Input(dim),
           ArgMax.Attrs().OutputType(DT_INT32)).asOutput()

fun TF_CPP.square(a: Output, name: String = "", scope: Scope = root) =
    Square(scope.WithOpName(name), Input(a)).asOutput()

fun TF_CPP.log(a: Output, name: String = "", scope: Scope = root) =
    Log(scope.WithOpName(name), Input(a)).asOutput()

fun TF_CPP.neg(a: Output, name: String = "", scope: Scope = root) =
    Negate(scope.WithOpName(name), Input(a)).asOutput()

fun TF_CPP.addN(vararg a: Output, name: String = "", scope: Scope = root) =
    AddN(scope.WithOpName(name), InputList(OutputVector(*a))).asOutput()