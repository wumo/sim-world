/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Output

interface gen_linalg_ops {
  fun cholesky(input: Output, name: String = "Cholesky") = run {
    buildOpTensor("Cholesky", name) {
      addInput(input, false)
    }
  }
  
  fun choleskyGrad(l: Output, grad: Output, name: String = "CholeskyGrad") = run {
    buildOpTensor("CholeskyGrad", name) {
      addInput(l, false)
      addInput(grad, false)
    }
  }
  
  fun logMatrixDeterminant(input: Output, name: String = "LogMatrixDeterminant") = run {
    buildOpTensors("LogMatrixDeterminant", name) {
      addInput(input, false)
    }
  }
  
  fun matrixDeterminant(input: Output, name: String = "MatrixDeterminant") = run {
    buildOpTensor("MatrixDeterminant", name) {
      addInput(input, false)
    }
  }
  
  fun matrixExponential(input: Output, name: String = "MatrixExponential") = run {
    buildOpTensor("MatrixExponential", name) {
      addInput(input, false)
    }
  }
  
  fun matrixInverse(input: Output, adjoint: Boolean = false, name: String = "MatrixInverse") = run {
    buildOpTensor("MatrixInverse", name) {
      addInput(input, false)
      attr("adjoint", adjoint)
    }
  }
  
  fun matrixSolve(matrix: Output, rhs: Output, adjoint: Boolean = false, name: String = "MatrixSolve") = run {
    buildOpTensor("MatrixSolve", name) {
      addInput(matrix, false)
      addInput(rhs, false)
      attr("adjoint", adjoint)
    }
  }
  
  fun matrixSolveLs(matrix: Output, rhs: Output, l2_regularizer: Output, fast: Boolean = true, name: String = "MatrixSolveLs") = run {
    buildOpTensor("MatrixSolveLs", name) {
      addInput(matrix, false)
      addInput(rhs, false)
      addInput(l2_regularizer, false)
      attr("fast", fast)
    }
  }
  
  fun matrixTriangularSolve(matrix: Output, rhs: Output, lower: Boolean = true, adjoint: Boolean = false, name: String = "MatrixTriangularSolve") = run {
    buildOpTensor("MatrixTriangularSolve", name) {
      addInput(matrix, false)
      addInput(rhs, false)
      attr("lower", lower)
      attr("adjoint", adjoint)
    }
  }
  
  fun qr(input: Output, full_matrices: Boolean = false, name: String = "Qr") = run {
    buildOpTensors("Qr", name) {
      addInput(input, false)
      attr("full_matrices", full_matrices)
    }
  }
  
  fun selfAdjointEigV2(input: Output, compute_v: Boolean = true, name: String = "SelfAdjointEigV2") = run {
    buildOpTensors("SelfAdjointEigV2", name) {
      addInput(input, false)
      attr("compute_v", compute_v)
    }
  }
  
  fun svd(input: Output, compute_uv: Boolean = true, full_matrices: Boolean = false, name: String = "Svd") = run {
    buildOpTensors("Svd", name) {
      addInput(input, false)
      attr("compute_uv", compute_uv)
      attr("full_matrices", full_matrices)
    }
  }
  
  fun matrixLogarithm(input: Output, name: String = "MatrixLogarithm") = run {
    buildOpTensor("MatrixLogarithm", name) {
      addInput(input, false)
    }
  }
}