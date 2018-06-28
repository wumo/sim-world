package wumo.sim.algorithm.util.c_api.math_ops

import wumo.sim.algorithm.util.c_api.Operation
import wumo.sim.algorithm.util.c_api.TF_C
import wumo.sim.algorithm.util.c_api.core.const

fun TF_C.unaryOp(op: String, a: Operation, name: String = op) =
    scope(name) {
      g.opBuilder(op, contextPath)
          .addInput(a[0])
          .build()
    }

fun TF_C.binaryOp(op: String, a: Operation, b: Operation, name: String = op) =
    scope(name) {
      g.opBuilder(op, contextPath)
          .addInput(a[0])
          .addInput(b[0])
          .build()
    }

fun TF_C.add(a: Operation, b: Operation, name: String = "add") =
    binaryOp("Add", a, b, name)

fun TF_C.sub(a: Operation, b: Operation, name: String = "sub") =
    binaryOp("Sub", a, b, name)

fun TF_C.mul(a: Operation, b: Operation, name: String = "mul") =
    binaryOp("Mul", a, b, name)

fun TF_C.div(a: Operation, b: Operation, name: String = "div") =
    binaryOp("Div", a, b, name)

fun TF_C.sum(a: Operation, b: Operation, name: String = "sum") =
    binaryOp("Sum", a, b, name)

fun TF_C.matmul(a: Operation, b: Operation, name: String = "matmul") =
    binaryOp("MatMul", a, b, name)

fun TF_C.argmax(a: Operation, dim: Int, name: String = "argmax") =
    scope(name) { binaryOp("ArgMax", a, const(dim, "dimension"), useContextName) }

fun TF_C.argmax(a: Operation, b: Operation, name: String = "argmax") =
    binaryOp("ArgMax", a, b, name)

fun TF_C.square(a: Operation, name: String = "square") =
    unaryOp("Square", a, name)