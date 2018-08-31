/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.types.*
import wumo.sim.util.Shape

interface gen_math_ops {
  fun abs(x: Output, name: String = "Abs") = run {
    buildOpTensor("Abs", name) {
      addInput(x, false)
    }
  }
  
  fun accumulateNV2(inputs: List<Output>, shape: Shape, name: String = "AccumulateNV2") = run {
    buildOpTensor("AccumulateNV2", name) {
      addInput(inputs, false)
      attr("shape", shape)
    }
  }
  
  fun acos(x: Output, name: String = "Acos") = run {
    buildOpTensor("Acos", name) {
      addInput(x, false)
    }
  }
  
  fun acosh(x: Output, name: String = "Acosh") = run {
    buildOpTensor("Acosh", name) {
      addInput(x, false)
    }
  }
  
  fun add(x: Output, y: Output, name: String = "Add") = run {
    buildOpTensor("Add", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun addN(inputs: List<Output>, name: String = "AddN") = run {
    buildOpTensor("AddN", name) {
      addInput(inputs, false)
    }
  }
  
  fun addV2(x: Output, y: Output, name: String = "AddV2") = run {
    buildOpTensor("AddV2", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun all(input: Output, reductionIndices: Output, keepDims: Boolean = false, name: String = "All") = run {
    buildOpTensor("All", name) {
      addInput(input, false)
      addInput(reductionIndices, false)
      attr("keep_dims", keepDims)
    }
  }
  
  fun angle(input: Output, tout: DataType<*> = FLOAT, name: String = "Angle") = run {
    buildOpTensor("Angle", name) {
      addInput(input, false)
      attr("Tout", tout)
    }
  }
  
  fun any(input: Output, reductionIndices: Output, keepDims: Boolean = false, name: String = "Any") = run {
    buildOpTensor("Any", name) {
      addInput(input, false)
      addInput(reductionIndices, false)
      attr("keep_dims", keepDims)
    }
  }
  
  fun approximateEqual(x: Output, y: Output, tolerance: Float = 1.0E-5f, name: String = "ApproximateEqual") = run {
    buildOpTensor("ApproximateEqual", name) {
      addInput(x, false)
      addInput(y, false)
      attr("tolerance", tolerance)
    }
  }
  
  fun argMax(input: Output, dimension: Output, outputType: DataType<*> = INT64, name: String = "ArgMax") = run {
    buildOpTensor("ArgMax", name) {
      addInput(input, false)
      addInput(dimension, false)
      attr("output_type", outputType)
    }
  }
  
  fun argMin(input: Output, dimension: Output, outputType: DataType<*> = INT64, name: String = "ArgMin") = run {
    buildOpTensor("ArgMin", name) {
      addInput(input, false)
      addInput(dimension, false)
      attr("output_type", outputType)
    }
  }
  
  fun asin(x: Output, name: String = "Asin") = run {
    buildOpTensor("Asin", name) {
      addInput(x, false)
    }
  }
  
  fun asinh(x: Output, name: String = "Asinh") = run {
    buildOpTensor("Asinh", name) {
      addInput(x, false)
    }
  }
  
  fun atan(x: Output, name: String = "Atan") = run {
    buildOpTensor("Atan", name) {
      addInput(x, false)
    }
  }
  
  fun atan2(y: Output, x: Output, name: String = "Atan2") = run {
    buildOpTensor("Atan2", name) {
      addInput(y, false)
      addInput(x, false)
    }
  }
  
  fun atanh(x: Output, name: String = "Atanh") = run {
    buildOpTensor("Atanh", name) {
      addInput(x, false)
    }
  }
  
  fun batchMatMul(x: Output, y: Output, adjX: Boolean = false, adjY: Boolean = false, name: String = "BatchMatMul") = run {
    buildOpTensor("BatchMatMul", name) {
      addInput(x, false)
      addInput(y, false)
      attr("adj_x", adjX)
      attr("adj_y", adjY)
    }
  }
  
  fun besselI0e(x: Output, name: String = "BesselI0e") = run {
    buildOpTensor("BesselI0e", name) {
      addInput(x, false)
    }
  }
  
  fun besselI1e(x: Output, name: String = "BesselI1e") = run {
    buildOpTensor("BesselI1e", name) {
      addInput(x, false)
    }
  }
  
  fun betainc(a: Output, b: Output, x: Output, name: String = "Betainc") = run {
    buildOpTensor("Betainc", name) {
      addInput(a, false)
      addInput(b, false)
      addInput(x, false)
    }
  }
  
  fun bincount(arr: Output, size: Output, weights: Output, name: String = "Bincount") = run {
    buildOpTensor("Bincount", name) {
      addInput(arr, false)
      addInput(size, false)
      addInput(weights, false)
    }
  }
  
  fun bucketize(input: Output, boundaries: Array<Float>, name: String = "Bucketize") = run {
    buildOpTensor("Bucketize", name) {
      addInput(input, false)
      attr("boundaries", boundaries)
    }
  }
  
  fun cast(x: Output, dstT: DataType<*>, truncate: Boolean = false, name: String = "Cast") = run {
    buildOpTensor("Cast", name) {
      addInput(x, false)
      attr("DstT", dstT)
      attr("Truncate", truncate)
    }
  }
  
  fun ceil(x: Output, name: String = "Ceil") = run {
    buildOpTensor("Ceil", name) {
      addInput(x, false)
    }
  }
  
  fun clipByValue(t: Output, clipValueMin: Output, clipValueMax: Output, name: String = "ClipByValue") = run {
    buildOpTensor("ClipByValue", name) {
      addInput(t, false)
      addInput(clipValueMin, false)
      addInput(clipValueMax, false)
    }
  }
  
  fun compareAndBitpack(input: Output, threshold: Output, name: String = "CompareAndBitpack") = run {
    buildOpTensor("CompareAndBitpack", name) {
      addInput(input, false)
      addInput(threshold, false)
    }
  }
  
  fun complex(real: Output, imag: Output, tout: DataType<*> = COMPLEX64, name: String = "Complex") = run {
    buildOpTensor("Complex", name) {
      addInput(real, false)
      addInput(imag, false)
      attr("Tout", tout)
    }
  }
  
  fun complexAbs(x: Output, tout: DataType<*> = FLOAT, name: String = "ComplexAbs") = run {
    buildOpTensor("ComplexAbs", name) {
      addInput(x, false)
      attr("Tout", tout)
    }
  }
  
  fun conj(input: Output, name: String = "Conj") = run {
    buildOpTensor("Conj", name) {
      addInput(input, false)
    }
  }
  
  fun cos(x: Output, name: String = "Cos") = run {
    buildOpTensor("Cos", name) {
      addInput(x, false)
    }
  }
  
  fun cosh(x: Output, name: String = "Cosh") = run {
    buildOpTensor("Cosh", name) {
      addInput(x, false)
    }
  }
  
  fun cross(a: Output, b: Output, name: String = "Cross") = run {
    buildOpTensor("Cross", name) {
      addInput(a, false)
      addInput(b, false)
    }
  }
  
  fun cumprod(x: Output, axis: Output, exclusive: Boolean = false, reverse: Boolean = false, name: String = "Cumprod") = run {
    buildOpTensor("Cumprod", name) {
      addInput(x, false)
      addInput(axis, false)
      attr("exclusive", exclusive)
      attr("reverse", reverse)
    }
  }
  
  fun cumsum(x: Output, axis: Output, exclusive: Boolean = false, reverse: Boolean = false, name: String = "Cumsum") = run {
    buildOpTensor("Cumsum", name) {
      addInput(x, false)
      addInput(axis, false)
      attr("exclusive", exclusive)
      attr("reverse", reverse)
    }
  }
  
  fun digamma(x: Output, name: String = "Digamma") = run {
    buildOpTensor("Digamma", name) {
      addInput(x, false)
    }
  }
  
  fun div(x: Output, y: Output, name: String = "Div") = run {
    buildOpTensor("Div", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun divNoNan(x: Output, y: Output, name: String = "DivNoNan") = run {
    buildOpTensor("DivNoNan", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun equal(x: Output, y: Output, name: String = "Equal") = run {
    buildOpTensor("Equal", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun erf(x: Output, name: String = "Erf") = run {
    buildOpTensor("Erf", name) {
      addInput(x, false)
    }
  }
  
  fun erfc(x: Output, name: String = "Erfc") = run {
    buildOpTensor("Erfc", name) {
      addInput(x, false)
    }
  }
  
  fun exp(x: Output, name: String = "Exp") = run {
    buildOpTensor("Exp", name) {
      addInput(x, false)
    }
  }
  
  fun expm1(x: Output, name: String = "Expm1") = run {
    buildOpTensor("Expm1", name) {
      addInput(x, false)
    }
  }
  
  fun floor(x: Output, name: String = "Floor") = run {
    buildOpTensor("Floor", name) {
      addInput(x, false)
    }
  }
  
  fun floorDiv(x: Output, y: Output, name: String = "FloorDiv") = run {
    buildOpTensor("FloorDiv", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun floorMod(x: Output, y: Output, name: String = "FloorMod") = run {
    buildOpTensor("FloorMod", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun greater(x: Output, y: Output, name: String = "Greater") = run {
    buildOpTensor("Greater", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun greaterEqual(x: Output, y: Output, name: String = "GreaterEqual") = run {
    buildOpTensor("GreaterEqual", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun histogramFixedWidth(values: Output, valueRange: Output, nbins: Output, dtype: DataType<*> = INT32, name: String = "HistogramFixedWidth") = run {
    buildOpTensor("HistogramFixedWidth", name) {
      addInput(values, false)
      addInput(valueRange, false)
      addInput(nbins, false)
      attr("dtype", dtype)
    }
  }
  
  fun igamma(a: Output, x: Output, name: String = "Igamma") = run {
    buildOpTensor("Igamma", name) {
      addInput(a, false)
      addInput(x, false)
    }
  }
  
  fun igammac(a: Output, x: Output, name: String = "Igammac") = run {
    buildOpTensor("Igammac", name) {
      addInput(a, false)
      addInput(x, false)
    }
  }
  
  fun imag(input: Output, tout: DataType<*> = FLOAT, name: String = "Imag") = run {
    buildOpTensor("Imag", name) {
      addInput(input, false)
      attr("Tout", tout)
    }
  }
  
  fun inv(x: Output, name: String = "Inv") = run {
    buildOpTensor("Inv", name) {
      addInput(x, false)
    }
  }
  
  fun isFinite(x: Output, name: String = "IsFinite") = run {
    buildOpTensor("IsFinite", name) {
      addInput(x, false)
    }
  }
  
  fun isInf(x: Output, name: String = "IsInf") = run {
    buildOpTensor("IsInf", name) {
      addInput(x, false)
    }
  }
  
  fun isNan(x: Output, name: String = "IsNan") = run {
    buildOpTensor("IsNan", name) {
      addInput(x, false)
    }
  }
  
  fun less(x: Output, y: Output, name: String = "Less") = run {
    buildOpTensor("Less", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun lessEqual(x: Output, y: Output, name: String = "LessEqual") = run {
    buildOpTensor("LessEqual", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun lgamma(x: Output, name: String = "Lgamma") = run {
    buildOpTensor("Lgamma", name) {
      addInput(x, false)
    }
  }
  
  fun linSpace(start: Output, stop: Output, num: Output, name: String = "LinSpace") = run {
    buildOpTensor("LinSpace", name) {
      addInput(start, false)
      addInput(stop, false)
      addInput(num, false)
    }
  }
  
  fun log(x: Output, name: String = "Log") = run {
    buildOpTensor("Log", name) {
      addInput(x, false)
    }
  }
  
  fun log1p(x: Output, name: String = "Log1p") = run {
    buildOpTensor("Log1p", name) {
      addInput(x, false)
    }
  }
  
  fun logicalAnd(x: Output, y: Output, name: String = "LogicalAnd") = run {
    buildOpTensor("LogicalAnd", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun logicalNot(x: Output, name: String = "LogicalNot") = run {
    buildOpTensor("LogicalNot", name) {
      addInput(x, false)
    }
  }
  
  fun logicalOr(x: Output, y: Output, name: String = "LogicalOr") = run {
    buildOpTensor("LogicalOr", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun matMul(a: Output, b: Output, transposeA: Boolean = false, transposeB: Boolean = false, name: String = "MatMul") = run {
    buildOpTensor("MatMul", name) {
      addInput(a, false)
      addInput(b, false)
      attr("transpose_a", transposeA)
      attr("transpose_b", transposeB)
    }
  }
  
  fun max(input: Output, reductionIndices: Output, keepDims: Boolean = false, name: String = "Max") = run {
    buildOpTensor("Max", name) {
      addInput(input, false)
      addInput(reductionIndices, false)
      attr("keep_dims", keepDims)
    }
  }
  
  fun maximum(x: Output, y: Output, name: String = "Maximum") = run {
    buildOpTensor("Maximum", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun mean(input: Output, reductionIndices: Output, keepDims: Boolean = false, name: String = "Mean") = run {
    buildOpTensor("Mean", name) {
      addInput(input, false)
      addInput(reductionIndices, false)
      attr("keep_dims", keepDims)
    }
  }
  
  fun min(input: Output, reductionIndices: Output, keepDims: Boolean = false, name: String = "Min") = run {
    buildOpTensor("Min", name) {
      addInput(input, false)
      addInput(reductionIndices, false)
      attr("keep_dims", keepDims)
    }
  }
  
  fun minimum(x: Output, y: Output, name: String = "Minimum") = run {
    buildOpTensor("Minimum", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun mod(x: Output, y: Output, name: String = "Mod") = run {
    buildOpTensor("Mod", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun mul(x: Output, y: Output, name: String = "Mul") = run {
    buildOpTensor("Mul", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun neg(x: Output, name: String = "Neg") = run {
    buildOpTensor("Neg", name) {
      addInput(x, false)
    }
  }
  
  fun notEqual(x: Output, y: Output, name: String = "NotEqual") = run {
    buildOpTensor("NotEqual", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun polygamma(a: Output, x: Output, name: String = "Polygamma") = run {
    buildOpTensor("Polygamma", name) {
      addInput(a, false)
      addInput(x, false)
    }
  }
  
  fun pow(x: Output, y: Output, name: String = "Pow") = run {
    buildOpTensor("Pow", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun _prod(input: Output, reductionIndices: Output, keepDims: Boolean = false, name: String = "Prod") = run {
    buildOpTensor("Prod", name) {
      addInput(input, false)
      addInput(reductionIndices, false)
      attr("keep_dims", keepDims)
    }
  }
  
  fun quantizeDownAndShrinkRange(input: Output, inputMin: Output, inputMax: Output, outType: DataType<*>, name: String = "QuantizeDownAndShrinkRange") = run {
    buildOpTensors("QuantizeDownAndShrinkRange", name) {
      addInput(input, false)
      addInput(inputMin, false)
      addInput(inputMax, false)
      attr("out_type", outType)
    }
  }
  
  fun quantizedAdd(x: Output, y: Output, minX: Output, maxX: Output, minY: Output, maxY: Output, toutput: DataType<*> = QINT32, name: String = "QuantizedAdd") = run {
    buildOpTensors("QuantizedAdd", name) {
      addInput(x, false)
      addInput(y, false)
      addInput(minX, false)
      addInput(maxX, false)
      addInput(minY, false)
      addInput(maxY, false)
      attr("Toutput", toutput)
    }
  }
  
  fun quantizedMatMul(a: Output, b: Output, minA: Output, maxA: Output, minB: Output, maxB: Output, toutput: DataType<*> = QINT32, transposeA: Boolean = false, transposeB: Boolean = false, tactivation: DataType<*> = QUINT8, name: String = "QuantizedMatMul") = run {
    buildOpTensors("QuantizedMatMul", name) {
      addInput(a, false)
      addInput(b, false)
      addInput(minA, false)
      addInput(maxA, false)
      addInput(minB, false)
      addInput(maxB, false)
      attr("Toutput", toutput)
      attr("transpose_a", transposeA)
      attr("transpose_b", transposeB)
      attr("Tactivation", tactivation)
    }
  }
  
  fun quantizedMul(x: Output, y: Output, minX: Output, maxX: Output, minY: Output, maxY: Output, toutput: DataType<*> = QINT32, name: String = "QuantizedMul") = run {
    buildOpTensors("QuantizedMul", name) {
      addInput(x, false)
      addInput(y, false)
      addInput(minX, false)
      addInput(maxX, false)
      addInput(minY, false)
      addInput(maxY, false)
      attr("Toutput", toutput)
    }
  }
  
  fun _range(start: Output, limit: Output, delta: Output, name: String = "Range") = run {
    buildOpTensor("Range", name) {
      addInput(start, false)
      addInput(limit, false)
      addInput(delta, false)
    }
  }
  
  fun real(input: Output, tout: DataType<*> = FLOAT, name: String = "Real") = run {
    buildOpTensor("Real", name) {
      addInput(input, false)
      attr("Tout", tout)
    }
  }
  
  fun realDiv(x: Output, y: Output, name: String = "RealDiv") = run {
    buildOpTensor("RealDiv", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun reciprocal(x: Output, name: String = "Reciprocal") = run {
    buildOpTensor("Reciprocal", name) {
      addInput(x, false)
    }
  }
  
  fun requantizationRange(input: Output, inputMin: Output, inputMax: Output, name: String = "RequantizationRange") = run {
    buildOpTensors("RequantizationRange", name) {
      addInput(input, false)
      addInput(inputMin, false)
      addInput(inputMax, false)
    }
  }
  
  fun requantize(input: Output, inputMin: Output, inputMax: Output, requestedOutputMin: Output, requestedOutputMax: Output, outType: DataType<*>, name: String = "Requantize") = run {
    buildOpTensors("Requantize", name) {
      addInput(input, false)
      addInput(inputMin, false)
      addInput(inputMax, false)
      addInput(requestedOutputMin, false)
      addInput(requestedOutputMax, false)
      attr("out_type", outType)
    }
  }
  
  fun rint(x: Output, name: String = "Rint") = run {
    buildOpTensor("Rint", name) {
      addInput(x, false)
    }
  }
  
  fun round(x: Output, name: String = "Round") = run {
    buildOpTensor("Round", name) {
      addInput(x, false)
    }
  }
  
  fun rsqrt(x: Output, name: String = "Rsqrt") = run {
    buildOpTensor("Rsqrt", name) {
      addInput(x, false)
    }
  }
  
  fun segmentMax(data: Output, segmentIds: Output, name: String = "SegmentMax") = run {
    buildOpTensor("SegmentMax", name) {
      addInput(data, false)
      addInput(segmentIds, false)
    }
  }
  
  fun segmentMean(data: Output, segmentIds: Output, name: String = "SegmentMean") = run {
    buildOpTensor("SegmentMean", name) {
      addInput(data, false)
      addInput(segmentIds, false)
    }
  }
  
  fun segmentMin(data: Output, segmentIds: Output, name: String = "SegmentMin") = run {
    buildOpTensor("SegmentMin", name) {
      addInput(data, false)
      addInput(segmentIds, false)
    }
  }
  
  fun segmentProd(data: Output, segmentIds: Output, name: String = "SegmentProd") = run {
    buildOpTensor("SegmentProd", name) {
      addInput(data, false)
      addInput(segmentIds, false)
    }
  }
  
  fun segmentSum(data: Output, segmentIds: Output, name: String = "SegmentSum") = run {
    buildOpTensor("SegmentSum", name) {
      addInput(data, false)
      addInput(segmentIds, false)
    }
  }
  
  fun select(condition: Output, t: Output, e: Output, name: String = "Select") = run {
    buildOpTensor("Select", name) {
      addInput(condition, false)
      addInput(t, false)
      addInput(e, false)
    }
  }
  
  fun sigmoid(x: Output, name: String = "Sigmoid") = run {
    buildOpTensor("Sigmoid", name) {
      addInput(x, false)
    }
  }
  
  fun sign(x: Output, name: String = "Sign") = run {
    buildOpTensor("Sign", name) {
      addInput(x, false)
    }
  }
  
  fun sin(x: Output, name: String = "Sin") = run {
    buildOpTensor("Sin", name) {
      addInput(x, false)
    }
  }
  
  fun sinh(x: Output, name: String = "Sinh") = run {
    buildOpTensor("Sinh", name) {
      addInput(x, false)
    }
  }
  
  fun sparseMatMul(a: Output, b: Output, transposeA: Boolean = false, transposeB: Boolean = false, aIsSparse: Boolean = false, bIsSparse: Boolean = false, name: String = "SparseMatMul") = run {
    buildOpTensor("SparseMatMul", name) {
      addInput(a, false)
      addInput(b, false)
      attr("transpose_a", transposeA)
      attr("transpose_b", transposeB)
      attr("a_is_sparse", aIsSparse)
      attr("b_is_sparse", bIsSparse)
    }
  }
  
  fun sparseSegmentMean(data: Output, indices: Output, segmentIds: Output, name: String = "SparseSegmentMean") = run {
    buildOpTensor("SparseSegmentMean", name) {
      addInput(data, false)
      addInput(indices, false)
      addInput(segmentIds, false)
    }
  }
  
  fun sparseSegmentMeanGrad(grad: Output, indices: Output, segmentIds: Output, outputDim0: Output, name: String = "SparseSegmentMeanGrad") = run {
    buildOpTensor("SparseSegmentMeanGrad", name) {
      addInput(grad, false)
      addInput(indices, false)
      addInput(segmentIds, false)
      addInput(outputDim0, false)
    }
  }
  
  fun sparseSegmentMeanWithNumSegments(data: Output, indices: Output, segmentIds: Output, numSegments: Output, name: String = "SparseSegmentMeanWithNumSegments") = run {
    buildOpTensor("SparseSegmentMeanWithNumSegments", name) {
      addInput(data, false)
      addInput(indices, false)
      addInput(segmentIds, false)
      addInput(numSegments, false)
    }
  }
  
  fun sparseSegmentSqrtN(data: Output, indices: Output, segmentIds: Output, name: String = "SparseSegmentSqrtN") = run {
    buildOpTensor("SparseSegmentSqrtN", name) {
      addInput(data, false)
      addInput(indices, false)
      addInput(segmentIds, false)
    }
  }
  
  fun sparseSegmentSqrtNGrad(grad: Output, indices: Output, segmentIds: Output, outputDim0: Output, name: String = "SparseSegmentSqrtNGrad") = run {
    buildOpTensor("SparseSegmentSqrtNGrad", name) {
      addInput(grad, false)
      addInput(indices, false)
      addInput(segmentIds, false)
      addInput(outputDim0, false)
    }
  }
  
  fun sparseSegmentSqrtNWithNumSegments(data: Output, indices: Output, segmentIds: Output, numSegments: Output, name: String = "SparseSegmentSqrtNWithNumSegments") = run {
    buildOpTensor("SparseSegmentSqrtNWithNumSegments", name) {
      addInput(data, false)
      addInput(indices, false)
      addInput(segmentIds, false)
      addInput(numSegments, false)
    }
  }
  
  fun sparseSegmentSum(data: Output, indices: Output, segmentIds: Output, name: String = "SparseSegmentSum") = run {
    buildOpTensor("SparseSegmentSum", name) {
      addInput(data, false)
      addInput(indices, false)
      addInput(segmentIds, false)
    }
  }
  
  fun sparseSegmentSumWithNumSegments(data: Output, indices: Output, segmentIds: Output, numSegments: Output, name: String = "SparseSegmentSumWithNumSegments") = run {
    buildOpTensor("SparseSegmentSumWithNumSegments", name) {
      addInput(data, false)
      addInput(indices, false)
      addInput(segmentIds, false)
      addInput(numSegments, false)
    }
  }
  
  fun sqrt(x: Output, name: String = "Sqrt") = run {
    buildOpTensor("Sqrt", name) {
      addInput(x, false)
    }
  }
  
  fun square(x: Output, name: String = "Square") = run {
    buildOpTensor("Square", name) {
      addInput(x, false)
    }
  }
  
  fun squaredDifference(x: Output, y: Output, name: String = "SquaredDifference") = run {
    buildOpTensor("SquaredDifference", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun sub(x: Output, y: Output, name: String = "Sub") = run {
    buildOpTensor("Sub", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun _sum(input: Output, reductionIndices: Output, keepDims: Boolean = false, name: String = "Sum") = run {
    buildOpTensor("Sum", name) {
      addInput(input, false)
      addInput(reductionIndices, false)
      attr("keep_dims", keepDims)
    }
  }
  
  fun tan(x: Output, name: String = "Tan") = run {
    buildOpTensor("Tan", name) {
      addInput(x, false)
    }
  }
  
  fun tanh(x: Output, name: String = "Tanh") = run {
    buildOpTensor("Tanh", name) {
      addInput(x, false)
    }
  }
  
  fun truncateDiv(x: Output, y: Output, name: String = "TruncateDiv") = run {
    buildOpTensor("TruncateDiv", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun truncateMod(x: Output, y: Output, name: String = "TruncateMod") = run {
    buildOpTensor("TruncateMod", name) {
      addInput(x, false)
      addInput(y, false)
    }
  }
  
  fun unsortedSegmentMax(data: Output, segmentIds: Output, numSegments: Output, name: String = "UnsortedSegmentMax") = run {
    buildOpTensor("UnsortedSegmentMax", name) {
      addInput(data, false)
      addInput(segmentIds, false)
      addInput(numSegments, false)
    }
  }
  
  fun unsortedSegmentMin(data: Output, segmentIds: Output, numSegments: Output, name: String = "UnsortedSegmentMin") = run {
    buildOpTensor("UnsortedSegmentMin", name) {
      addInput(data, false)
      addInput(segmentIds, false)
      addInput(numSegments, false)
    }
  }
  
  fun unsortedSegmentProd(data: Output, segmentIds: Output, numSegments: Output, name: String = "UnsortedSegmentProd") = run {
    buildOpTensor("UnsortedSegmentProd", name) {
      addInput(data, false)
      addInput(segmentIds, false)
      addInput(numSegments, false)
    }
  }
  
  fun unsortedSegmentSum(data: Output, segmentIds: Output, numSegments: Output, name: String = "UnsortedSegmentSum") = run {
    buildOpTensor("UnsortedSegmentSum", name) {
      addInput(data, false)
      addInput(segmentIds, false)
      addInput(numSegments, false)
    }
  }
  
  fun zeta(x: Output, q: Output, name: String = "Zeta") = run {
    buildOpTensor("Zeta", name) {
      addInput(x, false)
      addInput(q, false)
    }
  }
  
  fun igammaGradA(a: Output, x: Output, name: String = "IgammaGradA") = run {
    buildOpTensor("IgammaGradA", name) {
      addInput(a, false)
      addInput(x, false)
    }
  }
  
  fun invGrad(y: Output, dy: Output, name: String = "InvGrad") = run {
    buildOpTensor("InvGrad", name) {
      addInput(y, false)
      addInput(dy, false)
    }
  }
  
  fun reciprocalGrad(y: Output, dy: Output, name: String = "ReciprocalGrad") = run {
    buildOpTensor("ReciprocalGrad", name) {
      addInput(y, false)
      addInput(dy, false)
    }
  }
  
  fun rsqrtGrad(y: Output, dy: Output, name: String = "RsqrtGrad") = run {
    buildOpTensor("RsqrtGrad", name) {
      addInput(y, false)
      addInput(dy, false)
    }
  }
  
  fun sigmoidGrad(y: Output, dy: Output, name: String = "SigmoidGrad") = run {
    buildOpTensor("SigmoidGrad", name) {
      addInput(y, false)
      addInput(dy, false)
    }
  }
  
  fun sqrtGrad(y: Output, dy: Output, name: String = "SqrtGrad") = run {
    buildOpTensor("SqrtGrad", name) {
      addInput(y, false)
      addInput(dy, false)
    }
  }
  
  fun tanhGrad(y: Output, dy: Output, name: String = "TanhGrad") = run {
    buildOpTensor("TanhGrad", name) {
      addInput(y, false)
      addInput(dy, false)
    }
  }
}