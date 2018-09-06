package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_linalg_ops

object linalg_ops {
  interface API {
    fun batchCholesky(input: Output, name: String = "BatchCholesky"): Output {
      return gen_linalg_ops.batchCholesky(input, name)
    }
    
    fun batchCholeskyGrad(l: Output, grad: Output, name: String = "BatchCholeskyGrad"): Output {
      return gen_linalg_ops.batchCholeskyGrad(l, grad, name)
    }
    
    fun batchMatrixDeterminant(input: Output, name: String = "BatchMatrixDeterminant"): Output {
      return gen_linalg_ops.batchMatrixDeterminant(input, name)
    }
    
    fun batchMatrixInverse(input: Output, adjoint: Boolean = false, name: String = "BatchMatrixInverse"): Output {
      return gen_linalg_ops.batchMatrixInverse(input, adjoint, name)
    }
    
    fun batchMatrixSolve(matrix: Output, rhs: Output, adjoint: Boolean = false, name: String = "BatchMatrixSolve"): Output {
      return gen_linalg_ops.batchMatrixSolve(matrix, rhs, adjoint, name)
    }
    
    fun batchMatrixSolveLs(matrix: Output, rhs: Output, l2Regularizer: Output, fast: Boolean = true, name: String = "BatchMatrixSolveLs"): Output {
      return gen_linalg_ops.batchMatrixSolveLs(matrix, rhs, l2Regularizer, fast, name)
    }
    
    fun batchMatrixTriangularSolve(matrix: Output, rhs: Output, lower: Boolean = true, adjoint: Boolean = false, name: String = "BatchMatrixTriangularSolve"): Output {
      return gen_linalg_ops.batchMatrixTriangularSolve(matrix, rhs, lower, adjoint, name)
    }
    
    fun batchSelfAdjointEig(input: Output, name: String = "BatchSelfAdjointEig"): Output {
      return gen_linalg_ops.batchSelfAdjointEig(input, name)
    }
    
    fun batchSelfAdjointEigV2(input: Output, computeV: Boolean = true, name: String = "BatchSelfAdjointEigV2"): List<Output> {
      return gen_linalg_ops.batchSelfAdjointEigV2(input, computeV, name)
    }
    
    fun batchSvd(input: Output, computeUv: Boolean = true, fullMatrices: Boolean = false, name: String = "BatchSvd"): List<Output> {
      return gen_linalg_ops.batchSvd(input, computeUv, fullMatrices, name)
    }
    
    fun cholesky(input: Output, name: String = "Cholesky"): Output {
      return gen_linalg_ops.cholesky(input, name)
    }
    
    fun choleskyGrad(l: Output, grad: Output, name: String = "CholeskyGrad"): Output {
      return gen_linalg_ops.choleskyGrad(l, grad, name)
    }
    
    fun logMatrixDeterminant(input: Output, name: String = "LogMatrixDeterminant"): List<Output> {
      return gen_linalg_ops.logMatrixDeterminant(input, name)
    }
    
    fun matrixDeterminant(input: Output, name: String = "MatrixDeterminant"): Output {
      return gen_linalg_ops.matrixDeterminant(input, name)
    }
    
    fun matrixExponential(input: Output, name: String = "MatrixExponential"): Output {
      return gen_linalg_ops.matrixExponential(input, name)
    }
    
    fun matrixInverse(input: Output, adjoint: Boolean = false, name: String = "MatrixInverse"): Output {
      return gen_linalg_ops.matrixInverse(input, adjoint, name)
    }
    
    fun matrixLogarithm(input: Output, name: String = "MatrixLogarithm"): Output {
      return gen_linalg_ops.matrixLogarithm(input, name)
    }
    
    fun matrixSolve(matrix: Output, rhs: Output, adjoint: Boolean = false, name: String = "MatrixSolve"): Output {
      return gen_linalg_ops.matrixSolve(matrix, rhs, adjoint, name)
    }
    
    fun matrixSolveLs(matrix: Output, rhs: Output, l2Regularizer: Output, fast: Boolean = true, name: String = "MatrixSolveLs"): Output {
      return gen_linalg_ops.matrixSolveLs(matrix, rhs, l2Regularizer, fast, name)
    }
    
    fun matrixTriangularSolve(matrix: Output, rhs: Output, lower: Boolean = true, adjoint: Boolean = false, name: String = "MatrixTriangularSolve"): Output {
      return gen_linalg_ops.matrixTriangularSolve(matrix, rhs, lower, adjoint, name)
    }
    
    fun qr(input: Output, fullMatrices: Boolean = false, name: String = "Qr"): List<Output> {
      return gen_linalg_ops.qr(input, fullMatrices, name)
    }
    
    fun selfAdjointEig(input: Output, name: String = "SelfAdjointEig"): Output {
      return gen_linalg_ops.selfAdjointEig(input, name)
    }
    
    fun selfAdjointEigV2(input: Output, computeV: Boolean = true, name: String = "SelfAdjointEigV2"): List<Output> {
      return gen_linalg_ops.selfAdjointEigV2(input, computeV, name)
    }
    
    fun svd(input: Output, computeUv: Boolean = true, fullMatrices: Boolean = false, name: String = "Svd"): List<Output> {
      return gen_linalg_ops.svd(input, computeUv, fullMatrices, name)
    }
  }
}