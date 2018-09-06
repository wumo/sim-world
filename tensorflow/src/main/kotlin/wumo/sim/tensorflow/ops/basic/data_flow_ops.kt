package wumo.sim.tensorflow.ops.basic

import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.ops.gen.gen_data_flow_ops
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape

object data_flow_ops {
  interface API {
    fun accumulatorApplyGradient(handle: Output, localStep: Output, gradient: Output, name: String = "AccumulatorApplyGradient"): Op {
      return gen_data_flow_ops.accumulatorApplyGradient(handle, localStep, gradient, name)
    }
    
    fun accumulatorNumAccumulated(handle: Output, name: String = "AccumulatorNumAccumulated"): Output {
      return gen_data_flow_ops.accumulatorNumAccumulated(handle, name)
    }
    
    fun accumulatorSetGlobalStep(handle: Output, newGlobalStep: Output, name: String = "AccumulatorSetGlobalStep"): Op {
      return gen_data_flow_ops.accumulatorSetGlobalStep(handle, newGlobalStep, name)
    }
    
    fun accumulatorTakeGradient(handle: Output, numRequired: Output, dtype: DataType<*>, name: String = "AccumulatorTakeGradient"): Output {
      return gen_data_flow_ops.accumulatorTakeGradient(handle, numRequired, dtype, name)
    }
    
    fun barrier(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "Barrier"): Output {
      return gen_data_flow_ops.barrier(componentTypes, shapes, capacity, container, sharedName, name)
    }
    
    fun barrierClose(handle: Output, cancelPendingEnqueues: Boolean = false, name: String = "BarrierClose"): Op {
      return gen_data_flow_ops.barrierClose(handle, cancelPendingEnqueues, name)
    }
    
    fun barrierIncompleteSize(handle: Output, name: String = "BarrierIncompleteSize"): Output {
      return gen_data_flow_ops.barrierIncompleteSize(handle, name)
    }
    
    fun barrierInsertMany(handle: Output, keys: Output, values: Output, componentIndex: Long, name: String = "BarrierInsertMany"): Op {
      return gen_data_flow_ops.barrierInsertMany(handle, keys, values, componentIndex, name)
    }
    
    fun barrierReadySize(handle: Output, name: String = "BarrierReadySize"): Output {
      return gen_data_flow_ops.barrierReadySize(handle, name)
    }
    
    fun barrierTakeMany(handle: Output, numElements: Output, componentTypes: Array<Long>, allowSmallBatch: Boolean = false, waitForIncomplete: Boolean = false, timeoutMs: Long = -1L, name: String = "BarrierTakeMany"): List<Output> {
      return gen_data_flow_ops.barrierTakeMany(handle, numElements, componentTypes, allowSmallBatch, waitForIncomplete, timeoutMs, name)
    }
    
    fun conditionalAccumulator(dtype: DataType<*>, shape: Shape, container: String = "", sharedName: String = "", name: String = "ConditionalAccumulator"): Output {
      return gen_data_flow_ops.conditionalAccumulator(dtype, shape, container, sharedName, name)
    }
    
    fun deleteSessionTensor(handle: Output, name: String = "DeleteSessionTensor"): Op {
      return gen_data_flow_ops.deleteSessionTensor(handle, name)
    }
    
    fun dynamicPartition(data: Output, partitions: Output, numPartitions: Long, name: String = "DynamicPartition"): List<Output> {
      return gen_data_flow_ops.dynamicPartition(data, partitions, numPartitions, name)
    }
    
    fun dynamicStitch(indices: List<Output>, data: List<Output>, name: String = "DynamicStitch"): Output {
      return gen_data_flow_ops.dynamicStitch(indices, data, name)
    }
    
    fun fIFOQueue(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "FIFOQueue"): Output {
      return gen_data_flow_ops.fIFOQueue(componentTypes, shapes, capacity, container, sharedName, name)
    }
    
    fun fIFOQueueV2(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "FIFOQueueV2"): Output {
      return gen_data_flow_ops.fIFOQueueV2(componentTypes, shapes, capacity, container, sharedName, name)
    }
    
    fun fakeQueue(resource: Output, name: String = "FakeQueue"): Output {
      return gen_data_flow_ops.fakeQueue(resource, name)
    }
    
    fun getSessionHandle(value: Output, name: String = "GetSessionHandle"): Output {
      return gen_data_flow_ops.getSessionHandle(value, name)
    }
    
    fun getSessionHandleV2(value: Output, name: String = "GetSessionHandleV2"): Output {
      return gen_data_flow_ops.getSessionHandleV2(value, name)
    }
    
    fun getSessionTensor(handle: Output, dtype: DataType<*>, name: String = "GetSessionTensor"): Output {
      return gen_data_flow_ops.getSessionTensor(handle, dtype, name)
    }
    
    fun mapClear(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapClear"): Op {
      return gen_data_flow_ops.mapClear(dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun mapIncompleteSize(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapIncompleteSize"): Output {
      return gen_data_flow_ops.mapIncompleteSize(dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun mapPeek(key: Output, indices: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapPeek"): List<Output> {
      return gen_data_flow_ops.mapPeek(key, indices, dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun mapSize(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapSize"): Output {
      return gen_data_flow_ops.mapSize(dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun mapStage(key: Output, indices: Output, values: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapStage"): Op {
      return gen_data_flow_ops.mapStage(key, indices, values, dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun mapUnstage(key: Output, indices: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapUnstage"): List<Output> {
      return gen_data_flow_ops.mapUnstage(key, indices, dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun mapUnstageNoKey(indices: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapUnstageNoKey"): List<Output> {
      return gen_data_flow_ops.mapUnstageNoKey(indices, dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun orderedMapClear(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapClear"): Op {
      return gen_data_flow_ops.orderedMapClear(dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun orderedMapIncompleteSize(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapIncompleteSize"): Output {
      return gen_data_flow_ops.orderedMapIncompleteSize(dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun orderedMapPeek(key: Output, indices: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapPeek"): List<Output> {
      return gen_data_flow_ops.orderedMapPeek(key, indices, dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun orderedMapSize(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapSize"): Output {
      return gen_data_flow_ops.orderedMapSize(dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun orderedMapStage(key: Output, indices: Output, values: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapStage"): Op {
      return gen_data_flow_ops.orderedMapStage(key, indices, values, dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun orderedMapUnstage(key: Output, indices: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapUnstage"): List<Output> {
      return gen_data_flow_ops.orderedMapUnstage(key, indices, dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun orderedMapUnstageNoKey(indices: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapUnstageNoKey"): List<Output> {
      return gen_data_flow_ops.orderedMapUnstageNoKey(indices, dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun paddingFIFOQueue(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "PaddingFIFOQueue"): Output {
      return gen_data_flow_ops.paddingFIFOQueue(componentTypes, shapes, capacity, container, sharedName, name)
    }
    
    fun paddingFIFOQueueV2(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "PaddingFIFOQueueV2"): Output {
      return gen_data_flow_ops.paddingFIFOQueueV2(componentTypes, shapes, capacity, container, sharedName, name)
    }
    
    fun parallelDynamicStitch(indices: List<Output>, data: List<Output>, name: String = "ParallelDynamicStitch"): Output {
      return gen_data_flow_ops.parallelDynamicStitch(indices, data, name)
    }
    
    fun priorityQueue(shapes: Array<Shape>, componentTypes: Array<Long> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "PriorityQueue"): Output {
      return gen_data_flow_ops.priorityQueue(shapes, componentTypes, capacity, container, sharedName, name)
    }
    
    fun priorityQueueV2(shapes: Array<Shape>, componentTypes: Array<Long> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "PriorityQueueV2"): Output {
      return gen_data_flow_ops.priorityQueueV2(shapes, componentTypes, capacity, container, sharedName, name)
    }
    
    fun queueClose(handle: Output, cancelPendingEnqueues: Boolean = false, name: String = "QueueClose"): Op {
      return gen_data_flow_ops.queueClose(handle, cancelPendingEnqueues, name)
    }
    
    fun queueCloseV2(handle: Output, cancelPendingEnqueues: Boolean = false, name: String = "QueueCloseV2"): Op {
      return gen_data_flow_ops.queueCloseV2(handle, cancelPendingEnqueues, name)
    }
    
    fun queueDequeue(handle: Output, componentTypes: Array<Long>, timeoutMs: Long = -1L, name: String = "QueueDequeue"): List<Output> {
      return gen_data_flow_ops.queueDequeue(handle, componentTypes, timeoutMs, name)
    }
    
    fun queueDequeueMany(handle: Output, n: Output, componentTypes: Array<Long>, timeoutMs: Long = -1L, name: String = "QueueDequeueMany"): List<Output> {
      return gen_data_flow_ops.queueDequeueMany(handle, n, componentTypes, timeoutMs, name)
    }
    
    fun queueDequeueManyV2(handle: Output, n: Output, componentTypes: Array<Long>, timeoutMs: Long = -1L, name: String = "QueueDequeueManyV2"): List<Output> {
      return gen_data_flow_ops.queueDequeueManyV2(handle, n, componentTypes, timeoutMs, name)
    }
    
    fun queueDequeueUpTo(handle: Output, n: Output, componentTypes: Array<Long>, timeoutMs: Long = -1L, name: String = "QueueDequeueUpTo"): List<Output> {
      return gen_data_flow_ops.queueDequeueUpTo(handle, n, componentTypes, timeoutMs, name)
    }
    
    fun queueDequeueUpToV2(handle: Output, n: Output, componentTypes: Array<Long>, timeoutMs: Long = -1L, name: String = "QueueDequeueUpToV2"): List<Output> {
      return gen_data_flow_ops.queueDequeueUpToV2(handle, n, componentTypes, timeoutMs, name)
    }
    
    fun queueDequeueV2(handle: Output, componentTypes: Array<Long>, timeoutMs: Long = -1L, name: String = "QueueDequeueV2"): List<Output> {
      return gen_data_flow_ops.queueDequeueV2(handle, componentTypes, timeoutMs, name)
    }
    
    fun queueEnqueue(handle: Output, components: Output, timeoutMs: Long = -1L, name: String = "QueueEnqueue"): Op {
      return gen_data_flow_ops.queueEnqueue(handle, components, timeoutMs, name)
    }
    
    fun queueEnqueueMany(handle: Output, components: Output, timeoutMs: Long = -1L, name: String = "QueueEnqueueMany"): Op {
      return gen_data_flow_ops.queueEnqueueMany(handle, components, timeoutMs, name)
    }
    
    fun queueEnqueueManyV2(handle: Output, components: Output, timeoutMs: Long = -1L, name: String = "QueueEnqueueManyV2"): Op {
      return gen_data_flow_ops.queueEnqueueManyV2(handle, components, timeoutMs, name)
    }
    
    fun queueEnqueueV2(handle: Output, components: Output, timeoutMs: Long = -1L, name: String = "QueueEnqueueV2"): Op {
      return gen_data_flow_ops.queueEnqueueV2(handle, components, timeoutMs, name)
    }
    
    fun queueIsClosed(handle: Output, name: String = "QueueIsClosed"): Output {
      return gen_data_flow_ops.queueIsClosed(handle, name)
    }
    
    fun queueIsClosedV2(handle: Output, name: String = "QueueIsClosedV2"): Output {
      return gen_data_flow_ops.queueIsClosedV2(handle, name)
    }
    
    fun queueSize(handle: Output, name: String = "QueueSize"): Output {
      return gen_data_flow_ops.queueSize(handle, name)
    }
    
    fun queueSizeV2(handle: Output, name: String = "QueueSizeV2"): Output {
      return gen_data_flow_ops.queueSizeV2(handle, name)
    }
    
    fun randomShuffleQueue(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, minAfterDequeue: Long = 0L, seed: Long = 0L, seed2: Long = 0L, container: String = "", sharedName: String = "", name: String = "RandomShuffleQueue"): Output {
      return gen_data_flow_ops.randomShuffleQueue(componentTypes, shapes, capacity, minAfterDequeue, seed, seed2, container, sharedName, name)
    }
    
    fun randomShuffleQueueV2(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, minAfterDequeue: Long = 0L, seed: Long = 0L, seed2: Long = 0L, container: String = "", sharedName: String = "", name: String = "RandomShuffleQueueV2"): Output {
      return gen_data_flow_ops.randomShuffleQueueV2(componentTypes, shapes, capacity, minAfterDequeue, seed, seed2, container, sharedName, name)
    }
    
    fun recordInput(filePattern: String, fileRandomSeed: Long = 301L, fileShuffleShiftRatio: Float = 0.0f, fileBufferSize: Long = 10000L, fileParallelism: Long = 16L, batchSize: Long = 32L, compressionType: String = "", name: String = "RecordInput"): Output {
      return gen_data_flow_ops.recordInput(filePattern, fileRandomSeed, fileShuffleShiftRatio, fileBufferSize, fileParallelism, batchSize, compressionType, name)
    }
    
    fun sparseAccumulatorApplyGradient(handle: Output, localStep: Output, gradientIndices: Output, gradientValues: Output, gradientShape: Output, hasKnownShape: Boolean, name: String = "SparseAccumulatorApplyGradient"): Op {
      return gen_data_flow_ops.sparseAccumulatorApplyGradient(handle, localStep, gradientIndices, gradientValues, gradientShape, hasKnownShape, name)
    }
    
    fun sparseAccumulatorTakeGradient(handle: Output, numRequired: Output, dtype: DataType<*>, name: String = "SparseAccumulatorTakeGradient"): List<Output> {
      return gen_data_flow_ops.sparseAccumulatorTakeGradient(handle, numRequired, dtype, name)
    }
    
    fun sparseConditionalAccumulator(dtype: DataType<*>, shape: Shape, container: String = "", sharedName: String = "", name: String = "SparseConditionalAccumulator"): Output {
      return gen_data_flow_ops.sparseConditionalAccumulator(dtype, shape, container, sharedName, name)
    }
    
    fun stack(elemType: DataType<*>, stackName: String = "", name: String = "Stack"): Output {
      return gen_data_flow_ops.stack(elemType, stackName, name)
    }
    
    fun stackClose(handle: Output, name: String = "StackClose"): Op {
      return gen_data_flow_ops.stackClose(handle, name)
    }
    
    fun stackCloseV2(handle: Output, name: String = "StackCloseV2"): Op {
      return gen_data_flow_ops.stackCloseV2(handle, name)
    }
    
    fun stackPop(handle: Output, elemType: DataType<*>, name: String = "StackPop"): Output {
      return gen_data_flow_ops.stackPop(handle, elemType, name)
    }
    
    fun stackPopV2(handle: Output, elemType: DataType<*>, name: String = "StackPopV2"): Output {
      return gen_data_flow_ops.stackPopV2(handle, elemType, name)
    }
    
    fun stackPush(handle: Output, elem: Output, swapMemory: Boolean = false, name: String = "StackPush"): Output {
      return gen_data_flow_ops.stackPush(handle, elem, swapMemory, name)
    }
    
    fun stackPushV2(handle: Output, elem: Output, swapMemory: Boolean = false, name: String = "StackPushV2"): Output {
      return gen_data_flow_ops.stackPushV2(handle, elem, swapMemory, name)
    }
    
    fun stackV2(maxSize: Output, elemType: DataType<*>, stackName: String = "", name: String = "StackV2"): Output {
      return gen_data_flow_ops.stackV2(maxSize, elemType, stackName, name)
    }
    
    fun stage(values: Output, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "Stage"): Op {
      return gen_data_flow_ops.stage(values, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun stageClear(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "StageClear"): Op {
      return gen_data_flow_ops.stageClear(dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun stagePeek(index: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "StagePeek"): List<Output> {
      return gen_data_flow_ops.stagePeek(index, dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun stageSize(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "StageSize"): Output {
      return gen_data_flow_ops.stageSize(dtypes, capacity, memoryLimit, container, sharedName, name)
    }
    
    fun tensorArray(size: Output, dtype: DataType<*>, dynamicSize: Boolean = false, clearAfterRead: Boolean = true, tensorArrayName: String = "", elementShape: Shape = Shape(), name: String = "TensorArray"): Output {
      return gen_data_flow_ops.tensorArray(size, dtype, dynamicSize, clearAfterRead, tensorArrayName, elementShape, name)
    }
    
    fun tensorArrayClose(handle: Output, name: String = "TensorArrayClose"): Op {
      return gen_data_flow_ops.tensorArrayClose(handle, name)
    }
    
    fun tensorArrayCloseV2(handle: Output, name: String = "TensorArrayCloseV2"): Op {
      return gen_data_flow_ops.tensorArrayCloseV2(handle, name)
    }
    
    fun tensorArrayCloseV3(handle: Output, name: String = "TensorArrayCloseV3"): Op {
      return gen_data_flow_ops.tensorArrayCloseV3(handle, name)
    }
    
    fun tensorArrayConcat(handle: Output, flowIn: Output, dtype: DataType<*>, elementShapeExcept0: Shape = Shape(), name: String = "TensorArrayConcat"): List<Output> {
      return gen_data_flow_ops.tensorArrayConcat(handle, flowIn, dtype, elementShapeExcept0, name)
    }
    
    fun tensorArrayConcatV2(handle: Output, flowIn: Output, dtype: DataType<*>, elementShapeExcept0: Shape = Shape(), name: String = "TensorArrayConcatV2"): List<Output> {
      return gen_data_flow_ops.tensorArrayConcatV2(handle, flowIn, dtype, elementShapeExcept0, name)
    }
    
    fun tensorArrayConcatV3(handle: Output, flowIn: Output, dtype: DataType<*>, elementShapeExcept0: Shape = Shape(), name: String = "TensorArrayConcatV3"): List<Output> {
      return gen_data_flow_ops.tensorArrayConcatV3(handle, flowIn, dtype, elementShapeExcept0, name)
    }
    
    fun tensorArrayGather(handle: Output, indices: Output, flowIn: Output, dtype: DataType<*>, elementShape: Shape = Shape(), name: String = "TensorArrayGather"): Output {
      return gen_data_flow_ops.tensorArrayGather(handle, indices, flowIn, dtype, elementShape, name)
    }
    
    fun tensorArrayGatherV2(handle: Output, indices: Output, flowIn: Output, dtype: DataType<*>, elementShape: Shape = Shape(), name: String = "TensorArrayGatherV2"): Output {
      return gen_data_flow_ops.tensorArrayGatherV2(handle, indices, flowIn, dtype, elementShape, name)
    }
    
    fun tensorArrayGatherV3(handle: Output, indices: Output, flowIn: Output, dtype: DataType<*>, elementShape: Shape = Shape(), name: String = "TensorArrayGatherV3"): Output {
      return gen_data_flow_ops.tensorArrayGatherV3(handle, indices, flowIn, dtype, elementShape, name)
    }
    
    fun tensorArrayGrad(handle: Output, flowIn: Output, source: String, name: String = "TensorArrayGrad"): Output {
      return gen_data_flow_ops.tensorArrayGrad(handle, flowIn, source, name)
    }
    
    fun tensorArrayGradV2(handle: Output, flowIn: Output, source: String, name: String = "TensorArrayGradV2"): Output {
      return gen_data_flow_ops.tensorArrayGradV2(handle, flowIn, source, name)
    }
    
    fun tensorArrayGradV3(handle: Output, flowIn: Output, source: String, name: String = "TensorArrayGradV3"): List<Output> {
      return gen_data_flow_ops.tensorArrayGradV3(handle, flowIn, source, name)
    }
    
    fun tensorArrayGradWithShape(handle: Output, flowIn: Output, shapeToPrepend: Output, source: String, name: String = "TensorArrayGradWithShape"): List<Output> {
      return gen_data_flow_ops.tensorArrayGradWithShape(handle, flowIn, shapeToPrepend, source, name)
    }
    
    fun tensorArrayPack(handle: Output, flowIn: Output, dtype: DataType<*>, elementShape: Shape = Shape(), name: String = "TensorArrayPack"): Output {
      return gen_data_flow_ops.tensorArrayPack(handle, flowIn, dtype, elementShape, name)
    }
    
    fun tensorArrayRead(handle: Output, index: Output, flowIn: Output, dtype: DataType<*>, name: String = "TensorArrayRead"): Output {
      return gen_data_flow_ops.tensorArrayRead(handle, index, flowIn, dtype, name)
    }
    
    fun tensorArrayReadV2(handle: Output, index: Output, flowIn: Output, dtype: DataType<*>, name: String = "TensorArrayReadV2"): Output {
      return gen_data_flow_ops.tensorArrayReadV2(handle, index, flowIn, dtype, name)
    }
    
    fun tensorArrayReadV3(handle: Output, index: Output, flowIn: Output, dtype: DataType<*>, name: String = "TensorArrayReadV3"): Output {
      return gen_data_flow_ops.tensorArrayReadV3(handle, index, flowIn, dtype, name)
    }
    
    fun tensorArrayScatter(handle: Output, indices: Output, value: Output, flowIn: Output, name: String = "TensorArrayScatter"): Output {
      return gen_data_flow_ops.tensorArrayScatter(handle, indices, value, flowIn, name)
    }
    
    fun tensorArrayScatterV2(handle: Output, indices: Output, value: Output, flowIn: Output, name: String = "TensorArrayScatterV2"): Output {
      return gen_data_flow_ops.tensorArrayScatterV2(handle, indices, value, flowIn, name)
    }
    
    fun tensorArrayScatterV3(handle: Output, indices: Output, value: Output, flowIn: Output, name: String = "TensorArrayScatterV3"): Output {
      return gen_data_flow_ops.tensorArrayScatterV3(handle, indices, value, flowIn, name)
    }
    
    fun tensorArraySize(handle: Output, flowIn: Output, name: String = "TensorArraySize"): Output {
      return gen_data_flow_ops.tensorArraySize(handle, flowIn, name)
    }
    
    fun tensorArraySizeV2(handle: Output, flowIn: Output, name: String = "TensorArraySizeV2"): Output {
      return gen_data_flow_ops.tensorArraySizeV2(handle, flowIn, name)
    }
    
    fun tensorArraySizeV3(handle: Output, flowIn: Output, name: String = "TensorArraySizeV3"): Output {
      return gen_data_flow_ops.tensorArraySizeV3(handle, flowIn, name)
    }
    
    fun tensorArraySplit(handle: Output, value: Output, lengths: Output, flowIn: Output, name: String = "TensorArraySplit"): Output {
      return gen_data_flow_ops.tensorArraySplit(handle, value, lengths, flowIn, name)
    }
    
    fun tensorArraySplitV2(handle: Output, value: Output, lengths: Output, flowIn: Output, name: String = "TensorArraySplitV2"): Output {
      return gen_data_flow_ops.tensorArraySplitV2(handle, value, lengths, flowIn, name)
    }
    
    fun tensorArraySplitV3(handle: Output, value: Output, lengths: Output, flowIn: Output, name: String = "TensorArraySplitV3"): Output {
      return gen_data_flow_ops.tensorArraySplitV3(handle, value, lengths, flowIn, name)
    }
    
    fun tensorArrayUnpack(handle: Output, value: Output, flowIn: Output, name: String = "TensorArrayUnpack"): Output {
      return gen_data_flow_ops.tensorArrayUnpack(handle, value, flowIn, name)
    }
    
    fun tensorArrayV2(size: Output, dtype: DataType<*>, elementShape: Shape = Shape(), dynamicSize: Boolean = false, clearAfterRead: Boolean = true, tensorArrayName: String = "", name: String = "TensorArrayV2"): Output {
      return gen_data_flow_ops.tensorArrayV2(size, dtype, elementShape, dynamicSize, clearAfterRead, tensorArrayName, name)
    }
    
    fun tensorArrayV3(size: Output, dtype: DataType<*>, elementShape: Shape = Shape(), dynamicSize: Boolean = false, clearAfterRead: Boolean = true, identicalElementShapes: Boolean = false, tensorArrayName: String = "", name: String = "TensorArrayV3"): List<Output> {
      return gen_data_flow_ops.tensorArrayV3(size, dtype, elementShape, dynamicSize, clearAfterRead, identicalElementShapes, tensorArrayName, name)
    }
    
    fun tensorArrayWrite(handle: Output, index: Output, value: Output, flowIn: Output, name: String = "TensorArrayWrite"): Output {
      return gen_data_flow_ops.tensorArrayWrite(handle, index, value, flowIn, name)
    }
    
    fun tensorArrayWriteV2(handle: Output, index: Output, value: Output, flowIn: Output, name: String = "TensorArrayWriteV2"): Output {
      return gen_data_flow_ops.tensorArrayWriteV2(handle, index, value, flowIn, name)
    }
    
    fun tensorArrayWriteV3(handle: Output, index: Output, value: Output, flowIn: Output, name: String = "TensorArrayWriteV3"): Output {
      return gen_data_flow_ops.tensorArrayWriteV3(handle, index, value, flowIn, name)
    }
    
    fun unstage(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "Unstage"): List<Output> {
      return gen_data_flow_ops.unstage(dtypes, capacity, memoryLimit, container, sharedName, name)
    }
  }
}