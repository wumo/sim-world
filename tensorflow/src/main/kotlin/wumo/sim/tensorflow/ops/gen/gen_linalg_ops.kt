/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Output

object gen_linalg_ops {
  fun batchCholesky(input: Output, name: String = "BatchCholesky"): Output =
      buildOpTensor("BatchCholesky", name) {
        addInput(input, false)
      }
  
  fun batchCholeskyGrad(l: Output, grad: Output, name: String = "BatchCholeskyGrad"): Output =
      buildOpTensor("BatchCholeskyGrad", name) {
        addInput(l, false)
        addInput(grad, false)
      }
  
  fun batchMatrixDeterminant(input: Output, name: String = "BatchMatrixDeterminant"): Output =
      buildOpTensor("BatchMatrixDeterminant", name) {
        addInput(input, false)
      }
  
  fun batchMatrixInverse(input: Output, adjoint: Boolean = false, name: String = "BatchMatrixInverse"): Output =
      buildOpTensor("BatchMatrixInverse", name) {
        addInput(input, false)
        attr("adjoint", adjoint)
      }
  
  fun batchMatrixSolve(matrix: Output, rhs: Output, adjoint: Boolean = false, name: String = "BatchMatrixSolve"): Output =
      buildOpTensor("BatchMatrixSolve", name) {
        addInput(matrix, false)
        addInput(rhs, false)
        attr("adjoint", adjoint)
      }
  
  fun batchMatrixSolveLs(matrix: Output, rhs: Output, l2Regularizer: Output, fast: Boolean = true, name: String = "BatchMatrixSolveLs"): Output =
      buildOpTensor("BatchMatrixSolveLs", name) {
        addInput(matrix, false)
        addInput(rhs, false)
        addInput(l2Regularizer, false)
        attr("fast", fast)
      }
  
  fun batchMatrixTriangularSolve(matrix: Output, rhs: Output, lower: Boolean = true, adjoint: Boolean = false, name: String = "BatchMatrixTriangularSolve"): Output =
      buildOpTensor("BatchMatrixTriangularSolve", name) {
        addInput(matrix, false)
        addInput(rhs, false)
        attr("lower", lower)
        attr("adjoint", adjoint)
      }
  
  fun batchSelfAdjointEig(input: Output, name: String = "BatchSelfAdjointEig"): Output =
      buildOpTensor("BatchSelfAdjointEig", name) {
        addInput(input, false)
      }
  
  fun batchSelfAdjointEigV2(input: Output, computeV: Boolean = true, name: String = "BatchSelfAdjointEigV2"): List<Output> =
      buildOpTensors("BatchSelfAdjointEigV2", name) {
        addInput(input, false)
        attr("compute_v", computeV)
      }
  
  fun batchSvd(input: Output, computeUv: Boolean = true, fullMatrices: Boolean = false, name: String = "BatchSvd"): List<Output> =
      buildOpTensors("BatchSvd", name) {
        addInput(input, false)
        attr("compute_uv", computeUv)
        attr("full_matrices", fullMatrices)
      }
  
  fun cholesky(input: Output, name: String = "Cholesky"): Output =
      buildOpTensor("Cholesky", name) {
        addInput(input, false)
      }
  
  fun choleskyGrad(l: Output, grad: Output, name: String = "CholeskyGrad"): Output =
      buildOpTensor("CholeskyGrad", name) {
        addInput(l, false)
        addInput(grad, false)
      }
  
  fun logMatrixDeterminant(input: Output, name: String = "LogMatrixDeterminant"): List<Output> =
      buildOpTensors("LogMatrixDeterminant", name) {
        addInput(input, false)
      }
  
  fun matrixDeterminant(input: Output, name: String = "MatrixDeterminant"): Output =
      buildOpTensor("MatrixDeterminant", name) {
        addInput(input, false)
      }
  
  fun matrixExponential(input: Output, name: String = "MatrixExponential"): Output =
      buildOpTensor("MatrixExponential", name) {
        addInput(input, false)
      }
  
  fun matrixInverse(input: Output, adjoint: Boolean = false, name: String = "MatrixInverse"): Output =
      buildOpTensor("MatrixInverse", name) {
        addInput(input, false)
        attr("adjoint", adjoint)
      }
  
  fun matrixLogarithm(input: Output, name: String = "MatrixLogarithm"): Output =
      buildOpTensor("MatrixLogarithm", name) {
        addInput(input, false)
      }
  
  fun matrixSolve(matrix: Output, rhs: Output, adjoint: Boolean = false, name: String = "MatrixSolve"): Output =
      buildOpTensor("MatrixSolve", name) {
        addInput(matrix, false)
        addInput(rhs, false)
        attr("adjoint", adjoint)
      }
  
  fun matrixSolveLs(matrix: Output, rhs: Output, l2Regularizer: Output, fast: Boolean = true, name: String = "MatrixSolveLs"): Output =
      buildOpTensor("MatrixSolveLs", name) {
        addInput(matrix, false)
        addInput(rhs, false)
        addInput(l2Regularizer, false)
        attr("fast", fast)
      }
  
  fun matrixTriangularSolve(matrix: Output, rhs: Output, lower: Boolean = true, adjoint: Boolean = false, name: String = "MatrixTriangularSolve"): Output =
      buildOpTensor("MatrixTriangularSolve", name) {
        addInput(matrix, false)
        addInput(rhs, false)
        attr("lower", lower)
        attr("adjoint", adjoint)
      }
  
  fun qr(input: Output, fullMatrices: Boolean = false, name: String = "Qr"): List<Output> =
      buildOpTensors("Qr", name) {
        addInput(input, false)
        attr("full_matrices", fullMatrices)
      }
  
  fun selfAdjointEig(input: Output, name: String = "SelfAdjointEig"): Output =
      buildOpTensor("SelfAdjointEig", name) {
        addInput(input, false)
      }
  
  fun selfAdjointEigV2(input: Output, computeV: Boolean = true, name: String = "SelfAdjointEigV2"): List<Output> =
      buildOpTensors("SelfAdjointEigV2", name) {
        addInput(input, false)
        attr("compute_v", computeV)
      }
  
  fun svd(input: Output, computeUv: Boolean = true, fullMatrices: Boolean = false, name: String = "Svd"): List<Output> =
      buildOpTensors("Svd", name) {
        addInput(input, false)
        attr("compute_uv", computeUv)
        attr("full_matrices", fullMatrices)
      }
}