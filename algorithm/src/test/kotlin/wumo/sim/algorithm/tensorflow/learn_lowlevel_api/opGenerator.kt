package wumo.sim.algorithm.tensorflow.learn_lowlevel_api

import org.bytedeco.javacpp.Pointer
import org.bytedeco.javacpp.tensorflow
import org.tensorflow.framework.OpList
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.util.a

fun main(args: Array<String>) {
  TF
  generateFiles(ops)
}

val ops = mapOf(
    "Basic" to a(
        "ZerosLike", "OnesLike", "Fill", "Rank", "Size", "Shape", "ExpandDims", "Squeeze", "Pack", "ParallelConcat",
        "Unpack", "ConcatV2", "ConcatOffset", "Split", "SplitV", "Tile", "Pad", "PadV2", "MirrorPad", "Reshape",
        "Transpose", "ConjugateTranspose", "InvertPermutation", "ReverseV2", "ReverseSequence", "SpaceToBatchND",
        "BatchToSpaceND", "SpaceToDepth", "DepthToSpace", "Where", "Unique", "UniqueWithCounts", "ListDiff",
        "GatherV2", "GatherNd", "ScatterNd", "Slice", "StridedSlice", "CheckNumerics", "EditDistance", "OneHot",
        "BroadcastArgs", "StopGradient", "PreventGradient", "Identity", "IdentityN", "ScatterNdNonAliasingAdd",
        "QuantizeAndDequantizeV3", "QuantizeV2", "Dequantize", "QuantizedConcat", "QuantizedReshape",
        "QuantizedInstanceNorm", "FakeQuantWithMinMaxArgs", "FakeQuantWithMinMaxVars",
        "FakeQuantWithMinMaxVarsPerChannel"),
    "Math" to a(
        "Select", "Range", "LinSpace", "Cast", "Bitcast", "AddN", "Abs", "ComplexAbs", "Neg", "Reciprocal", "Square",
        "Sqrt", "Rsqrt", "Exp", "Expm1", "Log", "Log1p", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Sinh", "Cosh",
        "Tanh", "Asinh", "Acosh", "Atanh", "Lgamma", "Digamma", "Erf", "Erfc", "Sigmoid", "Sign", "Round", "Rint",
        "Floor", "Ceil", "IsNan", "IsInf", "IsFinite", "Add", "Sub", "Mul", "Div", "FloorDiv", "TruncateDiv",
        "RealDiv", "SquaredDifference", "Mod", "FloorMod", "TruncateMod", "Pow", "Igammac", "Igamma", "Zeta",
        "Polygamma", "Atan2", "Maximum", "Minimum", "Betainc", "LogicalNot", "LogicalAnd", "LogicalOr", "Equal",
        "NotEqual", "ApproximateEqual", "Less", "LessEqual", "Greater", "GreaterEqual", "Sum", "Mean", "Prod", "Min",
        "Max", "All", "Any", "ArgMax", "ArgMin", "Bincount", "Cumsum", "Cumprod", "SegmentSum", "SegmentMean",
        "SegmentProd", "SegmentMin", "SegmentMax", "UnsortedSegmentSum", "UnsortedSegmentMax",
        "SparseSegmentSum", "SparseSegmentMean", "SparseSegmentSqrtN",
        "SparseSegmentSumWithNumSegments", "SparseSegmentMeanWithNumSegments", "SparseSegmentSqrtNWithNumSegments",
        "Diag", "DiagPart", "MatrixDiag", "MatrixSetDiag",
        "MatrixDiagPart", "MatrixBandPart", "MatMul", "BatchMatMul", "SparseMatMul", "Cross", "Complex", "Real",
        "Imag", "Angle", "Conj", "Bucketize", "QuantizedAdd", "QuantizedMul", "QuantizedMatMul",
        "QuantizeDownAndShrinkRange", "Requantize", "RequantizationRange", "CompareAndBitpack"),
    "NN" to a(
        "BiasAdd", "Relu", "Relu6", "Elu", "Selu", "Softplus", "Softsign", "Softmax", "LogSoftmax", "L2Loss",
        "SoftmaxCrossEntropyWithLogits", "SparseSoftmaxCrossEntropyWithLogits", "TopKV2", "InTopKV2", "AvgPool",
        "AvgPool3D", "MaxPool", "MaxPoolGrad", "MaxPoolGradGrad", "MaxPool3D", "MaxPoolWithArgmax",
        "FractionalAvgPool", "FractionalMaxPool", "Conv2D", "Conv2DBackpropInput", "Conv2DBackpropFilter",
        "FusedResizeAndPadConv2D", "FusedPadConv2D", "DepthwiseConv2dNative", "Conv3D", "Dilation2D", "LRN",
        "BatchNormWithGlobalNormalization", "FusedBatchNorm", "QuantizedBiasAdd", "QuantizedRelu", "QuantizedRelu6",
        "QuantizedReluX", "QuantizedAvgPool", "QuantizedMaxPool", "QuantizedConv2D",
        "QuantizedBatchNormWithGlobalNormalization"),
    "Random" to a(
        "RandomShuffle", "RandomUniform", "RandomUniformInt", "RandomStandardNormal", "TruncatedNormal"),
    "Sparse" to a("SparseToDense"),
    "Text" to a(
        "StringJoin", "StringSplit", "EncodeBase64", "DecodeBase64", "StringToHashBucket", "StringToHashBucketFast",
        "StringToHashBucketStrong"))

fun generateFiles(ops: Map<String, Array<String>>) {
  val opdef = tensorflow.TF_GetAllOpList()
  val data = opdef.data()
  data.limit<Pointer>(opdef.length())
  val oplist = OpList.parseFrom(data.asByteBuffer()).opList
  val opDefsMap = oplist.map { it.name to it }.toMap()
}

