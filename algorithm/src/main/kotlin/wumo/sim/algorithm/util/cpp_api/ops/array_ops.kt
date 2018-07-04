package wumo.sim.algorithm.util.cpp_api.ops

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.util.cpp_api.TF_CPP

fun TF_CPP.zerosLike(x: Output, name: String = "", scope: Scope = root) =
    ZerosLike(scope.WithOpName(name), Input(x)).asOutput()

fun TF_CPP.onesLike(x: Output, name: String = "", scope: Scope = root) =
    OnesLike(scope.WithOpName(name), Input(x)).asOutput()

fun TF_CPP.slice(input: Output, begin: Output, size: Output, name: String = "", scope: Scope = root) =
    Slice(scope.WithOpName(name), Input(input), Input(begin), Input(size)).asOutput()