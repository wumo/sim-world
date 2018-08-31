import wumo.sim.util.a

val opGroups = mapOf(
    "array_ops" to a(
        "BatchToSpace", "BatchToSpaceND", "Bitcast", "BroadcastArgs", "BroadcastTo", "CheckNumerics", "ConcatV2", "ConjugateTranspose", "DebugGradientIdentity", "DebugGradientRefIdentity", "DeepCopy",
        "DepthToSpace", "Dequantize", "Diag", "DiagPart", "EditDistance", "Empty", "ExpandDims", "ExtractImagePatches", "FakeQuantWithMinMaxArgs", "FakeQuantWithMinMaxArgsGradient", "FakeQuantWithMinMaxVars",
        "FakeQuantWithMinMaxVarsGradient", "FakeQuantWithMinMaxVarsPerChannel", "FakeQuantWithMinMaxVarsPerChannelGradient", "Fill", "Gather", "GatherNd", "GatherV2", "GuaranteeConst", "Identity", "IdentityN", "ImmutableConst",
        "InplaceAdd", "InplaceSub", "InplaceUpdate", "InvertPermutation", "ListDiff", "MatrixBandPart", "MatrixDiag", "MatrixDiagPart", "MatrixSetDiag", "MirrorPad", "OneHot",
        "OnesLike", "Pack", "Pad", "PadV2", "ParallelConcat", "Placeholder", "PlaceholderWithDefault", "PreventGradient", "QuantizeAndDequantizeV2", "QuantizeAndDequantizeV3", "QuantizeV2",
        "QuantizedConcat", "QuantizedInstanceNorm", "QuantizedReshape", "Rank", "Reshape", "ResourceStridedSliceAssign", "ReverseSequence", "ReverseV2", "ScatterNd", "ScatterNdNonAliasingAdd", "Shape",
        "ShapeN", "Size", "Slice", "Snapshot", "SpaceToBatch", "SpaceToBatchND", "SpaceToDepth", "Split", "SplitV", "Squeeze", "StopGradient",
        "StridedSlice", "StridedSliceAssign", "StridedSliceGrad", "Tile", "Transpose", "Unique", "UniqueV2", "UniqueWithCounts", "UniqueWithCountsV2", "Unpack", "UnravelIndex",
        "Where", "ZerosLike", "BroadcastGradientArgs", "MirrorPadGrad", "RefIdentity"),
    "audio_ops" to a(
        "AudioSpectrogram", "DecodeWav", "EncodeWav", "Mfcc"),
    "batch_ops" to a(
        "Batch", "BatchFunction", "Unbatch", "UnbatchGrad"),
    "bitwise_ops" to a(
        "BitwiseAnd", "BitwiseOr", "BitwiseXor", "Invert", "LeftShift", "PopulationCount", "RightShift"),
    "boosted_trees_ops" to a(
        "BoostedTreesCalculateBestGainsPerFeature", "BoostedTreesCenterBias", "BoostedTreesCreateEnsemble", "BoostedTreesDeserializeEnsemble", "BoostedTreesEnsembleResourceHandleOp", "BoostedTreesExampleDebugOutputs", "BoostedTreesGetEnsembleStates", "BoostedTreesMakeStatsSummary", "BoostedTreesPredict", "BoostedTreesSerializeEnsemble", "BoostedTreesTrainingPredict",
        "BoostedTreesUpdateEnsemble", "IsBoostedTreesEnsembleInitialized"),
    "candidate_sampling_ops" to a(
        "AllCandidateSampler", "ComputeAccidentalHits", "FixedUnigramCandidateSampler", "LearnedUnigramCandidateSampler", "LogUniformCandidateSampler", "UniformCandidateSampler"),
    "checkpoint_ops" to a(
        "GenerateVocabRemapping", "LoadAndRemapMatrix"),
    "control_flow_ops" to a(
        "Abort", "ControlTrigger", "LoopCond", "Merge", "NextIteration", "RefNextIteration", "RefSelect", "RefSwitch", "Switch", "Enter", "Exit",
        "RefEnter", "RefExit", "RefMerge"),
    "ctc_ops" to a(
        "CTCBeamSearchDecoder", "CTCGreedyDecoder", "CTCLoss"),
    "cudnn_rnn_ops" to a(
        "CudnnRNN", "CudnnRNNBackprop", "CudnnRNNCanonicalToParams", "CudnnRNNParamsSize", "CudnnRNNParamsToCanonical", "CudnnRNNBackpropV2", "CudnnRNNV2"),
    "dataset_ops" to a(
        "AnonymousIterator", "BatchDataset", "BytesProducedStatsDataset", "CacheDataset", "ConcatenateDataset", "DatasetToSingleElement", "DenseToSparseBatchDataset", "DeserializeIterator", "EnqueueInQueueDataset", "FeatureStatsDataset", "FilterDataset",
        "FixedLengthRecordDataset", "FlatMapDataset", "GeneratorDataset", "GroupByWindowDataset", "InterleaveDataset", "Iterator", "IteratorFromStringHandle", "IteratorGetNext", "IteratorGetNextSync", "IteratorToStringHandle", "LatencyStatsDataset",
        "MakeIterator", "MapDataset", "OneShotIterator", "PaddedBatchDataset", "ParallelInterleaveDataset", "ParallelMapDataset", "PrefetchDataset", "PrependFromQueueAndPaddedBatchDataset", "RandomDataset", "RangeDataset", "RepeatDataset",
        "ScanDataset", "SerializeIterator", "SetStatsAggregatorDataset", "ShuffleAndRepeatDataset", "ShuffleDataset", "SkipDataset", "SlideDataset", "SparseTensorSliceDataset", "SqlDataset", "StatsAggregatorHandle", "StatsAggregatorSummary",
        "TFRecordDataset", "TakeDataset", "TensorDataset", "TensorSliceDataset", "TextLineDataset", "UnbatchDataset", "ZipDataset", "BatchDatasetV2", "DatasetToGraph", "DatasetToTFRecord", "GroupByReducerDataset",
        "IteratorFromStringHandleV2", "IteratorV2", "MapAndBatchDataset", "MapAndBatchDatasetV2", "OptimizeDataset", "PaddedBatchDatasetV2", "SinkDataset", "WindowDataset"),
    "data_flow_ops" to a(
        "AccumulatorApplyGradient", "AccumulatorNumAccumulated", "AccumulatorSetGlobalStep", "AccumulatorTakeGradient", "Barrier", "BarrierClose", "BarrierIncompleteSize", "BarrierInsertMany", "BarrierReadySize", "BarrierTakeMany", "ConditionalAccumulator",
        "DeleteSessionTensor", "DynamicPartition", "DynamicStitch", "FIFOQueueV2", "GetSessionHandle", "GetSessionHandleV2", "GetSessionTensor", "MapClear", "MapIncompleteSize", "MapPeek", "MapSize",
        "MapStage", "MapUnstage", "MapUnstageNoKey", "OrderedMapClear", "OrderedMapIncompleteSize", "OrderedMapPeek", "OrderedMapSize", "OrderedMapStage", "OrderedMapUnstage", "OrderedMapUnstageNoKey", "PaddingFIFOQueueV2",
        "ParallelDynamicStitch", "PriorityQueueV2", "QueueCloseV2", "QueueDequeueManyV2", "QueueDequeueUpToV2", "QueueDequeueV2", "QueueEnqueueManyV2", "QueueEnqueueV2", "QueueIsClosed", "QueueIsClosedV2", "QueueSizeV2",
        "RandomShuffleQueueV2", "RecordInput", "SparseAccumulatorApplyGradient", "SparseAccumulatorTakeGradient", "SparseConditionalAccumulator", "Stack", "StackClose", "StackCloseV2", "StackPop", "StackPopV2", "StackPush",
        "StackPushV2", "StackV2", "Stage", "StageClear", "StagePeek", "StageSize", "TensorArrayCloseV3", "TensorArrayConcatV3",
        "TensorArrayGatherV3", "TensorArrayGradV3", "TensorArrayGradWithShape", "TensorArrayReadV3", "TensorArrayScatterV3", "TensorArraySizeV3", "TensorArraySplitV3", "TensorArrayV3", "TensorArrayWriteV3", "Unstage"),
    "functional_ops" to a(
        "For", "If", "PartitionedCall", "RemoteCall", "StatefulPartitionedCall", "SymbolicGradient", "While"),
    "image_ops" to a(
        "AdjustContrastv2", "AdjustHue", "AdjustSaturation", "CropAndResize", "CropAndResizeGradBoxes", "CropAndResizeGradImage", "DecodeAndCropJpeg", "DecodeBmp", "DecodeGif", "DecodeJpeg", "DecodePng",
        "DrawBoundingBoxes", "EncodeJpeg", "EncodePng", "ExtractGlimpse", "ExtractJpegShape", "HSVToRGB", "NonMaxSuppression", "NonMaxSuppressionV2", "NonMaxSuppressionV3", "NonMaxSuppressionWithOverlaps", "QuantizedResizeBilinear",
        "RGBToHSV", "ResizeArea", "ResizeBicubic", "ResizeBilinear", "ResizeNearestNeighbor", "SampleDistortedBoundingBox", "SampleDistortedBoundingBoxV2", "ResizeBicubicGrad", "ResizeBilinearGrad", "ResizeNearestNeighborGrad"),
    "io_ops" to a(
        "FixedLengthRecordReaderV2", "IdentityReaderV2", "LMDBReader", "MatchingFiles", "MergeV2Checkpoints", "ReadFile", "ReaderNumRecordsProducedV2", "ReaderNumWorkUnitsCompletedV2", "ReaderReadUpToV2", "ReaderReadV2", "ReaderResetV2",
        "ReaderRestoreStateV2", "ReaderSerializeStateV2", "Restore", "RestoreSlice", "RestoreV2", "Save", "SaveSlices", "SaveV2", "ShardedFilename", "ShardedFilespec", "TFRecordReaderV2",
        "TextLineReaderV2", "WholeFileReaderV2", "WriteFile"),
    "linalg_ops" to a(
        "Cholesky", "CholeskyGrad", "LogMatrixDeterminant", "MatrixDeterminant", "MatrixExponential", "MatrixInverse", "MatrixSolve", "MatrixSolveLs", "MatrixTriangularSolve", "Qr", "SelfAdjointEigV2",
        "Svd", "MatrixLogarithm"),
    "list_ops" to a(
        "EmptyTensorList", "TensorListConcatLists", "TensorListElementShape", "TensorListFromTensor", "TensorListGetItem", "TensorListLength", "TensorListPopBack", "TensorListPushBack", "TensorListPushBackBatch", "TensorListReserve", "TensorListSetItem",
        "TensorListStack"),
    "logging_ops" to a(
        "Assert", "AudioSummaryV2", "HistogramSummary", "ImageSummary", "MergeSummary", "Print", "ScalarSummary", "TensorSummary", "TensorSummaryV2", "Timestamp"),
    "lookup_ops" to a(
        "HashTableV2", "InitializeTableFromTextFileV2", "InitializeTableV2", "LookupTableExportV2", "LookupTableFindV2", "LookupTableImportV2", "LookupTableInsertV2", "LookupTableSizeV2", "MutableDenseHashTableV2", "MutableHashTableOfTensorsV2", "MutableHashTableV2"),
    "manip_ops" to a(
        "Roll"),
    "math_ops" to a(
        "Abs", "AccumulateNV2", "Acos", "Acosh", "Add", "AddN", "AddV2", "All", "Angle", "Any", "ApproximateEqual",
        "ArgMax", "ArgMin", "Asin", "Asinh", "Atan", "Atan2", "Atanh", "BatchMatMul", "BesselI0e", "BesselI1e", "Betainc",
        "Bincount", "Bucketize", "Cast", "Ceil", "ClipByValue", "CompareAndBitpack", "Complex", "ComplexAbs", "Conj", "Cos", "Cosh",
        "Cross", "Cumprod", "Cumsum", "Digamma", "Div", "DivNoNan", "Equal", "Erf", "Erfc", "Exp", "Expm1", "Floor",
        "FloorDiv", "FloorMod", "Greater", "GreaterEqual", "HistogramFixedWidth", "Igamma", "Igammac", "Imag", "Inv", "IsFinite", "IsInf",
        "IsNan", "Less", "LessEqual", "Lgamma", "LinSpace", "Log", "Log1p", "LogicalAnd", "LogicalNot", "LogicalOr", "MatMul",
        "Max", "Maximum", "Mean", "Min", "Minimum", "Mod", "Mul", "Neg", "NotEqual", "Polygamma", "Pow",
        "Prod", "QuantizeDownAndShrinkRange", "QuantizedAdd", "QuantizedMatMul", "QuantizedMul",
        "Range", "Real", "RealDiv", "Reciprocal", "RequantizationRange", "Requantize",
        "Rint", "Round", "Rsqrt", "SegmentMax", "SegmentMean", "SegmentMin", "SegmentProd", "SegmentSum", "Select", "Sigmoid", "Sign",
        "Sin", "Sinh", "SparseMatMul", "SparseSegmentMean", "SparseSegmentMeanGrad", "SparseSegmentMeanWithNumSegments", "SparseSegmentSqrtN", "SparseSegmentSqrtNGrad", "SparseSegmentSqrtNWithNumSegments", "SparseSegmentSum", "SparseSegmentSumWithNumSegments",
        "Sqrt", "Square", "SquaredDifference", "Sub", "Sum", "Tan", "Tanh", "TruncateDiv", "TruncateMod", "UnsortedSegmentMax", "UnsortedSegmentMin",
        "UnsortedSegmentProd", "UnsortedSegmentSum", "Zeta", "IgammaGradA", "InvGrad", "ReciprocalGrad", "RsqrtGrad", "SigmoidGrad", "SqrtGrad", "TanhGrad"),
    "nn_ops" to a(
        "AvgPool", "AvgPool3D", "AvgPool3DGrad", "BiasAdd", "BiasAddGrad", "Conv2D", "Conv2DBackpropFilter", "Conv2DBackpropInput", "Conv3D", "Conv3DBackpropFilterV2", "Conv3DBackpropInputV2",
        "DataFormatDimMap", "DataFormatVecPermute", "DepthwiseConv2dNative", "DepthwiseConv2dNativeBackpropFilter", "DepthwiseConv2dNativeBackpropInput", "Dilation2D", "Dilation2DBackpropFilter", "Dilation2DBackpropInput", "Elu", "FractionalAvgPool", "FractionalMaxPool",
        "FusedBatchNorm", "FusedBatchNormGrad", "FusedBatchNormGradV2", "FusedBatchNormV2", "FusedPadConv2D", "FusedResizeAndPadConv2D", "InTopK", "InTopKV2", "L2Loss", "LRN", "LogSoftmax",
        "MaxPool", "MaxPool3D", "MaxPool3DGrad", "MaxPool3DGradGrad", "MaxPoolGradGrad", "MaxPoolGradGradV2", "MaxPoolGradGradWithArgmax", "MaxPoolGradV2", "MaxPoolV2", "MaxPoolWithArgmax", "NthElement",
        "QuantizedAvgPool", "QuantizedBatchNormWithGlobalNormalization", "QuantizedBiasAdd", "QuantizedConv2D", "QuantizedMaxPool", "QuantizedRelu", "QuantizedRelu6", "QuantizedReluX", "Relu", "Relu6", "Selu",
        "Softmax", "SoftmaxCrossEntropyWithLogits", "Softplus", "Softsign", "SparseSoftmaxCrossEntropyWithLogits", "TopKV2", "AvgPoolGrad", "EluGrad", "FractionalAvgPoolGrad", "FractionalMaxPoolGrad", "LRNGrad",
        "MaxPoolGrad", "MaxPoolGradWithArgmax", "Relu6Grad", "ReluGrad", "SeluGrad", "SoftplusGrad", "SoftsignGrad"),
    "no_op" to a(
        "NoOp"),
    "parsing_ops" to a(
        "DecodeCSV", "DecodeCompressed", "DecodeJSONExample", "DecodeRaw", "ParseExample", "ParseSingleExample", "ParseSingleSequenceExample", "ParseTensor", "SerializeTensor", "StringToNumber"),
    "random_ops" to a(
        "Multinomial", "ParameterizedTruncatedNormal", "RandomGamma", "RandomPoissonV2", "RandomShuffle", "RandomStandardNormal", "RandomUniform", "RandomUniformInt", "TruncatedNormal", "RandomGammaGrad"),
    "resource_variable_ops" to a(
        "AssignAddVariableOp", "AssignSubVariableOp", "AssignVariableOp", "ConsumeMutexLock", "DestroyResourceOp", "MutexLock", "MutexV2", "ReadVariableOp", "ResourceGather", "ResourceScatterAdd", "ResourceScatterDiv",
        "ResourceScatterMax", "ResourceScatterMin", "ResourceScatterMul", "ResourceScatterSub", "ResourceScatterUpdate", "VarHandleOp", "VarIsInitializedOp", "VariableShape"),
    "script_ops" to a(
        "EagerPyFunc"),
    "sdca_ops" to a(
        "SdcaFprint", "SdcaOptimizer", "SdcaShrinkL1"),
    "set_ops" to a(
        "DenseToDenseSetOperation", "DenseToSparseSetOperation", "SetSize", "SparseToSparseSetOperation"),
    "sparse_ops" to a(
        "AddManySparseToTensorsMap", "AddSparseToTensorsMap", "DeserializeManySparse", "DeserializeSparse", "SerializeManySparse", "SerializeSparse", "SparseAdd", "SparseAddGrad", "SparseConcat", "SparseCross", "SparseDenseCwiseAdd",
        "SparseDenseCwiseDiv", "SparseDenseCwiseMul", "SparseFillEmptyRows", "SparseFillEmptyRowsGrad", "SparseReduceMax", "SparseReduceMaxSparse", "SparseReduceSum", "SparseReduceSumSparse", "SparseReorder", "SparseReshape", "SparseSlice",
        "SparseSliceGrad", "SparseSoftmax", "SparseSparseMaximum", "SparseSparseMinimum", "SparseSplit", "SparseTensorDenseAdd", "SparseTensorDenseMatMul", "SparseToDense", "TakeManySparseFromTensorsMap"),
    "spectral_ops" to a(
        "FFT", "FFT2D", "FFT3D", "IFFT", "IFFT2D", "IFFT3D", "IRFFT", "IRFFT2D", "IRFFT3D", "RFFT", "RFFT2D",
        "RFFT3D"),
    "stateless_random_ops" to a(
        "StatelessMultinomial", "StatelessRandomNormal", "StatelessRandomUniform", "StatelessTruncatedNormal"),
    "state_ops" to a(
        "Assign", "AssignAdd", "AssignSub", "CountUpTo", "DestroyTemporaryVariable", "IsVariableInitialized", "ResourceCountUpTo", "ResourceScatterNdAdd", "ResourceScatterNdUpdate", "ScatterAdd", "ScatterDiv",
        "ScatterMax", "ScatterMin", "ScatterMul", "ScatterNdAdd", "ScatterNdSub", "ScatterNdUpdate", "ScatterSub", "ScatterUpdate", "TemporaryVariable", "VariableV2"),
    "string_ops" to a(
        "AsString", "DecodeBase64", "EncodeBase64", "ReduceJoin", "RegexFullMatch", "RegexReplace", "StringJoin", "StringSplit", "StringSplitV2", "StringStrip", "StringToHashBucket",
        "StringToHashBucketFast", "StringToHashBucketStrong", "Substr"),
    "summary_ops" to a(
        "CloseSummaryWriter", "CreateSummaryDbWriter", "CreateSummaryFileWriter", "FlushSummaryWriter", "ImportEvent", "SummaryWriter", "WriteAudioSummary", "WriteGraphSummary", "WriteHistogramSummary", "WriteImageSummary", "WriteScalarSummary",
        "WriteSummary"),
    "training_ops" to a(
        "ApplyAdadelta", "ApplyAdagrad", "ApplyAdagradDA", "ApplyAdam", "ApplyAddSign", "ApplyCenteredRMSProp", "ApplyFtrl", "ApplyFtrlV2", "ApplyGradientDescent", "ApplyMomentum", "ApplyPowerSign",
        "ApplyProximalAdagrad", "ApplyProximalGradientDescent", "ApplyRMSProp", "ResourceApplyAdadelta", "ResourceApplyAdagrad", "ResourceApplyAdagradDA", "ResourceApplyAdam", "ResourceApplyAddSign", "ResourceApplyCenteredRMSProp", "ResourceApplyFtrl", "ResourceApplyFtrlV2",
        "ResourceApplyGradientDescent", "ResourceApplyMomentum", "ResourceApplyPowerSign", "ResourceApplyProximalAdagrad", "ResourceApplyProximalGradientDescent", "ResourceApplyRMSProp", "ResourceSparseApplyAdadelta", "ResourceSparseApplyAdagrad", "ResourceSparseApplyAdagradDA", "ResourceSparseApplyCenteredRMSProp", "ResourceSparseApplyFtrl",
        "ResourceSparseApplyFtrlV2", "ResourceSparseApplyMomentum", "ResourceSparseApplyProximalAdagrad", "ResourceSparseApplyProximalGradientDescent", "ResourceSparseApplyRMSProp", "SparseApplyAdadelta", "SparseApplyAdagrad", "SparseApplyAdagradDA", "SparseApplyCenteredRMSProp", "SparseApplyFtrl", "SparseApplyFtrlV2",
        "SparseApplyMomentum", "SparseApplyProximalAdagrad", "SparseApplyProximalGradientDescent", "SparseApplyRMSProp", "ApplyAdaMax", "ResourceApplyAdaMax"),
    "user_ops" to a(
        "Fact"))

val renames = setOf("Gather", "OneHot", "Prod", "Range", "Sum")