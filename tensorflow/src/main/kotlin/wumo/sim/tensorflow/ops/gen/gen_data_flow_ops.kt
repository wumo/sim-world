/**
 * DO NOT EDIT THIS FILE - it is machine generated
 */
package wumo.sim.tensorflow.ops.gen

import wumo.sim.tensorflow.buildOp
import wumo.sim.tensorflow.buildOpTensor
import wumo.sim.tensorflow.buildOpTensors
import wumo.sim.tensorflow.ops.Output
import wumo.sim.tensorflow.types.DataType
import wumo.sim.util.Shape

interface gen_data_flow_ops {
  fun accumulatorApplyGradient(handle: Output, localStep: Output, gradient: Output, name: String = "AccumulatorApplyGradient") = run {
    buildOp("AccumulatorApplyGradient", name) {
      addInput(handle, true)
      addInput(localStep, false)
      addInput(gradient, false)
    }
  }
  
  fun accumulatorNumAccumulated(handle: Output, name: String = "AccumulatorNumAccumulated") = run {
    buildOpTensor("AccumulatorNumAccumulated", name) {
      addInput(handle, true)
    }
  }
  
  fun accumulatorSetGlobalStep(handle: Output, newGlobalStep: Output, name: String = "AccumulatorSetGlobalStep") = run {
    buildOp("AccumulatorSetGlobalStep", name) {
      addInput(handle, true)
      addInput(newGlobalStep, false)
    }
  }
  
  fun accumulatorTakeGradient(handle: Output, numRequired: Output, dtype: DataType<*>, name: String = "AccumulatorTakeGradient") = run {
    buildOpTensor("AccumulatorTakeGradient", name) {
      addInput(handle, true)
      addInput(numRequired, false)
      attr("dtype", dtype)
    }
  }
  
  fun barrier(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "Barrier") = run {
    buildOpTensor("Barrier", name) {
      attr("component_types", componentTypes)
      attr("shapes", shapes)
      attr("capacity", capacity)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun barrierClose(handle: Output, cancelPendingEnqueues: Boolean = false, name: String = "BarrierClose") = run {
    buildOp("BarrierClose", name) {
      addInput(handle, true)
      attr("cancel_pending_enqueues", cancelPendingEnqueues)
    }
  }
  
  fun barrierIncompleteSize(handle: Output, name: String = "BarrierIncompleteSize") = run {
    buildOpTensor("BarrierIncompleteSize", name) {
      addInput(handle, true)
    }
  }
  
  fun barrierInsertMany(handle: Output, keys: Output, values: Output, componentIndex: Long, name: String = "BarrierInsertMany") = run {
    buildOp("BarrierInsertMany", name) {
      addInput(handle, true)
      addInput(keys, false)
      addInput(values, false)
      attr("component_index", componentIndex)
    }
  }
  
  fun barrierReadySize(handle: Output, name: String = "BarrierReadySize") = run {
    buildOpTensor("BarrierReadySize", name) {
      addInput(handle, true)
    }
  }
  
  fun barrierTakeMany(handle: Output, numElements: Output, componentTypes: Array<Long>, allowSmallBatch: Boolean = false, waitForIncomplete: Boolean = false, timeoutMs: Long = -1L, name: String = "BarrierTakeMany") = run {
    buildOpTensors("BarrierTakeMany", name) {
      addInput(handle, true)
      addInput(numElements, false)
      attr("component_types", componentTypes)
      attr("allow_small_batch", allowSmallBatch)
      attr("wait_for_incomplete", waitForIncomplete)
      attr("timeout_ms", timeoutMs)
    }
  }
  
  fun conditionalAccumulator(dtype: DataType<*>, shape: Shape, container: String = "", sharedName: String = "", name: String = "ConditionalAccumulator") = run {
    buildOpTensor("ConditionalAccumulator", name) {
      attr("dtype", dtype)
      attr("shape", shape)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun deleteSessionTensor(handle: Output, name: String = "DeleteSessionTensor") = run {
    buildOp("DeleteSessionTensor", name) {
      addInput(handle, false)
    }
  }
  
  fun dynamicPartition(data: Output, partitions: Output, numPartitions: Long, name: String = "DynamicPartition") = run {
    buildOpTensors("DynamicPartition", name) {
      addInput(data, false)
      addInput(partitions, false)
      attr("num_partitions", numPartitions)
    }
  }
  
  fun dynamicStitch(indices: List<Output>, data: List<Output>, name: String = "DynamicStitch") = run {
    buildOpTensor("DynamicStitch", name) {
      addInput(indices, false)
      addInput(data, false)
    }
  }
  
  fun fIFOQueue(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "FIFOQueue") = run {
    buildOpTensor("FIFOQueue", name) {
      attr("component_types", componentTypes)
      attr("shapes", shapes)
      attr("capacity", capacity)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun fIFOQueueV2(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "FIFOQueueV2") = run {
    buildOpTensor("FIFOQueueV2", name) {
      attr("component_types", componentTypes)
      attr("shapes", shapes)
      attr("capacity", capacity)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun fakeQueue(resource: Output, name: String = "FakeQueue") = run {
    buildOpTensor("FakeQueue", name) {
      addInput(resource, false)
    }
  }
  
  fun getSessionHandle(value: Output, name: String = "GetSessionHandle") = run {
    buildOpTensor("GetSessionHandle", name) {
      addInput(value, false)
    }
  }
  
  fun getSessionHandleV2(value: Output, name: String = "GetSessionHandleV2") = run {
    buildOpTensor("GetSessionHandleV2", name) {
      addInput(value, false)
    }
  }
  
  fun getSessionTensor(handle: Output, dtype: DataType<*>, name: String = "GetSessionTensor") = run {
    buildOpTensor("GetSessionTensor", name) {
      addInput(handle, false)
      attr("dtype", dtype)
    }
  }
  
  fun mapClear(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapClear") = run {
    buildOp("MapClear", name) {
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun mapIncompleteSize(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapIncompleteSize") = run {
    buildOpTensor("MapIncompleteSize", name) {
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun mapPeek(key: Output, indices: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapPeek") = run {
    buildOpTensors("MapPeek", name) {
      addInput(key, false)
      addInput(indices, false)
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun mapSize(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapSize") = run {
    buildOpTensor("MapSize", name) {
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun mapStage(key: Output, indices: Output, values: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapStage") = run {
    buildOp("MapStage", name) {
      addInput(key, false)
      addInput(indices, false)
      addInput(values, false)
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun mapUnstage(key: Output, indices: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapUnstage") = run {
    buildOpTensors("MapUnstage", name) {
      addInput(key, false)
      addInput(indices, false)
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun mapUnstageNoKey(indices: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "MapUnstageNoKey") = run {
    buildOpTensors("MapUnstageNoKey", name) {
      addInput(indices, false)
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun orderedMapClear(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapClear") = run {
    buildOp("OrderedMapClear", name) {
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun orderedMapIncompleteSize(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapIncompleteSize") = run {
    buildOpTensor("OrderedMapIncompleteSize", name) {
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun orderedMapPeek(key: Output, indices: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapPeek") = run {
    buildOpTensors("OrderedMapPeek", name) {
      addInput(key, false)
      addInput(indices, false)
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun orderedMapSize(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapSize") = run {
    buildOpTensor("OrderedMapSize", name) {
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun orderedMapStage(key: Output, indices: Output, values: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapStage") = run {
    buildOp("OrderedMapStage", name) {
      addInput(key, false)
      addInput(indices, false)
      addInput(values, false)
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun orderedMapUnstage(key: Output, indices: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapUnstage") = run {
    buildOpTensors("OrderedMapUnstage", name) {
      addInput(key, false)
      addInput(indices, false)
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun orderedMapUnstageNoKey(indices: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "OrderedMapUnstageNoKey") = run {
    buildOpTensors("OrderedMapUnstageNoKey", name) {
      addInput(indices, false)
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun paddingFIFOQueue(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "PaddingFIFOQueue") = run {
    buildOpTensor("PaddingFIFOQueue", name) {
      attr("component_types", componentTypes)
      attr("shapes", shapes)
      attr("capacity", capacity)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun paddingFIFOQueueV2(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "PaddingFIFOQueueV2") = run {
    buildOpTensor("PaddingFIFOQueueV2", name) {
      attr("component_types", componentTypes)
      attr("shapes", shapes)
      attr("capacity", capacity)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun parallelDynamicStitch(indices: List<Output>, data: List<Output>, name: String = "ParallelDynamicStitch") = run {
    buildOpTensor("ParallelDynamicStitch", name) {
      addInput(indices, false)
      addInput(data, false)
    }
  }
  
  fun priorityQueue(shapes: Array<Shape>, componentTypes: Array<Long> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "PriorityQueue") = run {
    buildOpTensor("PriorityQueue", name) {
      attr("shapes", shapes)
      attr("component_types", componentTypes)
      attr("capacity", capacity)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun priorityQueueV2(shapes: Array<Shape>, componentTypes: Array<Long> = arrayOf(), capacity: Long = -1L, container: String = "", sharedName: String = "", name: String = "PriorityQueueV2") = run {
    buildOpTensor("PriorityQueueV2", name) {
      attr("shapes", shapes)
      attr("component_types", componentTypes)
      attr("capacity", capacity)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun queueClose(handle: Output, cancelPendingEnqueues: Boolean = false, name: String = "QueueClose") = run {
    buildOp("QueueClose", name) {
      addInput(handle, true)
      attr("cancel_pending_enqueues", cancelPendingEnqueues)
    }
  }
  
  fun queueCloseV2(handle: Output, cancelPendingEnqueues: Boolean = false, name: String = "QueueCloseV2") = run {
    buildOp("QueueCloseV2", name) {
      addInput(handle, false)
      attr("cancel_pending_enqueues", cancelPendingEnqueues)
    }
  }
  
  fun queueDequeue(handle: Output, componentTypes: Array<Long>, timeoutMs: Long = -1L, name: String = "QueueDequeue") = run {
    buildOpTensors("QueueDequeue", name) {
      addInput(handle, true)
      attr("component_types", componentTypes)
      attr("timeout_ms", timeoutMs)
    }
  }
  
  fun queueDequeueMany(handle: Output, n: Output, componentTypes: Array<Long>, timeoutMs: Long = -1L, name: String = "QueueDequeueMany") = run {
    buildOpTensors("QueueDequeueMany", name) {
      addInput(handle, true)
      addInput(n, false)
      attr("component_types", componentTypes)
      attr("timeout_ms", timeoutMs)
    }
  }
  
  fun queueDequeueManyV2(handle: Output, n: Output, componentTypes: Array<Long>, timeoutMs: Long = -1L, name: String = "QueueDequeueManyV2") = run {
    buildOpTensors("QueueDequeueManyV2", name) {
      addInput(handle, false)
      addInput(n, false)
      attr("component_types", componentTypes)
      attr("timeout_ms", timeoutMs)
    }
  }
  
  fun queueDequeueUpTo(handle: Output, n: Output, componentTypes: Array<Long>, timeoutMs: Long = -1L, name: String = "QueueDequeueUpTo") = run {
    buildOpTensors("QueueDequeueUpTo", name) {
      addInput(handle, true)
      addInput(n, false)
      attr("component_types", componentTypes)
      attr("timeout_ms", timeoutMs)
    }
  }
  
  fun queueDequeueUpToV2(handle: Output, n: Output, componentTypes: Array<Long>, timeoutMs: Long = -1L, name: String = "QueueDequeueUpToV2") = run {
    buildOpTensors("QueueDequeueUpToV2", name) {
      addInput(handle, false)
      addInput(n, false)
      attr("component_types", componentTypes)
      attr("timeout_ms", timeoutMs)
    }
  }
  
  fun queueDequeueV2(handle: Output, componentTypes: Array<Long>, timeoutMs: Long = -1L, name: String = "QueueDequeueV2") = run {
    buildOpTensors("QueueDequeueV2", name) {
      addInput(handle, false)
      attr("component_types", componentTypes)
      attr("timeout_ms", timeoutMs)
    }
  }
  
  fun queueEnqueue(handle: Output, components: Output, timeoutMs: Long = -1L, name: String = "QueueEnqueue") = run {
    buildOp("QueueEnqueue", name) {
      addInput(handle, true)
      addInput(components, false)
      attr("timeout_ms", timeoutMs)
    }
  }
  
  fun queueEnqueueMany(handle: Output, components: Output, timeoutMs: Long = -1L, name: String = "QueueEnqueueMany") = run {
    buildOp("QueueEnqueueMany", name) {
      addInput(handle, true)
      addInput(components, false)
      attr("timeout_ms", timeoutMs)
    }
  }
  
  fun queueEnqueueManyV2(handle: Output, components: Output, timeoutMs: Long = -1L, name: String = "QueueEnqueueManyV2") = run {
    buildOp("QueueEnqueueManyV2", name) {
      addInput(handle, false)
      addInput(components, false)
      attr("timeout_ms", timeoutMs)
    }
  }
  
  fun queueEnqueueV2(handle: Output, components: Output, timeoutMs: Long = -1L, name: String = "QueueEnqueueV2") = run {
    buildOp("QueueEnqueueV2", name) {
      addInput(handle, false)
      addInput(components, false)
      attr("timeout_ms", timeoutMs)
    }
  }
  
  fun queueIsClosed(handle: Output, name: String = "QueueIsClosed") = run {
    buildOpTensor("QueueIsClosed", name) {
      addInput(handle, true)
    }
  }
  
  fun queueIsClosedV2(handle: Output, name: String = "QueueIsClosedV2") = run {
    buildOpTensor("QueueIsClosedV2", name) {
      addInput(handle, false)
    }
  }
  
  fun queueSize(handle: Output, name: String = "QueueSize") = run {
    buildOpTensor("QueueSize", name) {
      addInput(handle, true)
    }
  }
  
  fun queueSizeV2(handle: Output, name: String = "QueueSizeV2") = run {
    buildOpTensor("QueueSizeV2", name) {
      addInput(handle, false)
    }
  }
  
  fun randomShuffleQueue(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, minAfterDequeue: Long = 0L, seed: Long = 0L, seed2: Long = 0L, container: String = "", sharedName: String = "", name: String = "RandomShuffleQueue") = run {
    buildOpTensor("RandomShuffleQueue", name) {
      attr("component_types", componentTypes)
      attr("shapes", shapes)
      attr("capacity", capacity)
      attr("min_after_dequeue", minAfterDequeue)
      attr("seed", seed)
      attr("seed2", seed2)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun randomShuffleQueueV2(componentTypes: Array<Long>, shapes: Array<Shape> = arrayOf(), capacity: Long = -1L, minAfterDequeue: Long = 0L, seed: Long = 0L, seed2: Long = 0L, container: String = "", sharedName: String = "", name: String = "RandomShuffleQueueV2") = run {
    buildOpTensor("RandomShuffleQueueV2", name) {
      attr("component_types", componentTypes)
      attr("shapes", shapes)
      attr("capacity", capacity)
      attr("min_after_dequeue", minAfterDequeue)
      attr("seed", seed)
      attr("seed2", seed2)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun recordInput(filePattern: String, fileRandomSeed: Long = 301L, fileShuffleShiftRatio: Float = 0.0f, fileBufferSize: Long = 10000L, fileParallelism: Long = 16L, batchSize: Long = 32L, compressionType: String = "", name: String = "RecordInput") = run {
    buildOpTensor("RecordInput", name) {
      attr("file_pattern", filePattern)
      attr("file_random_seed", fileRandomSeed)
      attr("file_shuffle_shift_ratio", fileShuffleShiftRatio)
      attr("file_buffer_size", fileBufferSize)
      attr("file_parallelism", fileParallelism)
      attr("batch_size", batchSize)
      attr("compression_type", compressionType)
    }
  }
  
  fun sparseAccumulatorApplyGradient(handle: Output, localStep: Output, gradientIndices: Output, gradientValues: Output, gradientShape: Output, hasKnownShape: Boolean, name: String = "SparseAccumulatorApplyGradient") = run {
    buildOp("SparseAccumulatorApplyGradient", name) {
      addInput(handle, true)
      addInput(localStep, false)
      addInput(gradientIndices, false)
      addInput(gradientValues, false)
      addInput(gradientShape, false)
      attr("has_known_shape", hasKnownShape)
    }
  }
  
  fun sparseAccumulatorTakeGradient(handle: Output, numRequired: Output, dtype: DataType<*>, name: String = "SparseAccumulatorTakeGradient") = run {
    buildOpTensors("SparseAccumulatorTakeGradient", name) {
      addInput(handle, true)
      addInput(numRequired, false)
      attr("dtype", dtype)
    }
  }
  
  fun sparseConditionalAccumulator(dtype: DataType<*>, shape: Shape, container: String = "", sharedName: String = "", name: String = "SparseConditionalAccumulator") = run {
    buildOpTensor("SparseConditionalAccumulator", name) {
      attr("dtype", dtype)
      attr("shape", shape)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun stack(elemType: DataType<*>, stackName: String = "", name: String = "Stack") = run {
    buildOpTensor("Stack", name) {
      attr("elem_type", elemType)
      attr("stack_name", stackName)
    }
  }
  
  fun stackClose(handle: Output, name: String = "StackClose") = run {
    buildOp("StackClose", name) {
      addInput(handle, true)
    }
  }
  
  fun stackCloseV2(handle: Output, name: String = "StackCloseV2") = run {
    buildOp("StackCloseV2", name) {
      addInput(handle, false)
    }
  }
  
  fun stackPop(handle: Output, elemType: DataType<*>, name: String = "StackPop") = run {
    buildOpTensor("StackPop", name) {
      addInput(handle, true)
      attr("elem_type", elemType)
    }
  }
  
  fun stackPopV2(handle: Output, elemType: DataType<*>, name: String = "StackPopV2") = run {
    buildOpTensor("StackPopV2", name) {
      addInput(handle, false)
      attr("elem_type", elemType)
    }
  }
  
  fun stackPush(handle: Output, elem: Output, swapMemory: Boolean = false, name: String = "StackPush") = run {
    buildOpTensor("StackPush", name) {
      addInput(handle, true)
      addInput(elem, false)
      attr("swap_memory", swapMemory)
    }
  }
  
  fun stackPushV2(handle: Output, elem: Output, swapMemory: Boolean = false, name: String = "StackPushV2") = run {
    buildOpTensor("StackPushV2", name) {
      addInput(handle, false)
      addInput(elem, false)
      attr("swap_memory", swapMemory)
    }
  }
  
  fun stackV2(maxSize: Output, elemType: DataType<*>, stackName: String = "", name: String = "StackV2") = run {
    buildOpTensor("StackV2", name) {
      addInput(maxSize, false)
      attr("elem_type", elemType)
      attr("stack_name", stackName)
    }
  }
  
  fun stage(values: Output, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "Stage") = run {
    buildOp("Stage", name) {
      addInput(values, false)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun stageClear(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "StageClear") = run {
    buildOp("StageClear", name) {
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun stagePeek(index: Output, dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "StagePeek") = run {
    buildOpTensors("StagePeek", name) {
      addInput(index, false)
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun stageSize(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "StageSize") = run {
    buildOpTensor("StageSize", name) {
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
  
  fun tensorArray(size: Output, dtype: DataType<*>, dynamicSize: Boolean = false, clearAfterRead: Boolean = true, tensorArrayName: String = "", elementShape: Shape = Shape(), name: String = "TensorArray") = run {
    buildOpTensor("TensorArray", name) {
      addInput(size, false)
      attr("dtype", dtype)
      attr("dynamic_size", dynamicSize)
      attr("clear_after_read", clearAfterRead)
      attr("tensor_array_name", tensorArrayName)
      attr("element_shape", elementShape)
    }
  }
  
  fun tensorArrayClose(handle: Output, name: String = "TensorArrayClose") = run {
    buildOp("TensorArrayClose", name) {
      addInput(handle, true)
    }
  }
  
  fun tensorArrayCloseV2(handle: Output, name: String = "TensorArrayCloseV2") = run {
    buildOp("TensorArrayCloseV2", name) {
      addInput(handle, false)
    }
  }
  
  fun tensorArrayCloseV3(handle: Output, name: String = "TensorArrayCloseV3") = run {
    buildOp("TensorArrayCloseV3", name) {
      addInput(handle, false)
    }
  }
  
  fun tensorArrayConcat(handle: Output, flowIn: Output, dtype: DataType<*>, elementShapeExcept0: Shape = Shape(), name: String = "TensorArrayConcat") = run {
    buildOpTensors("TensorArrayConcat", name) {
      addInput(handle, true)
      addInput(flowIn, false)
      attr("dtype", dtype)
      attr("element_shape_except0", elementShapeExcept0)
    }
  }
  
  fun tensorArrayConcatV2(handle: Output, flowIn: Output, dtype: DataType<*>, elementShapeExcept0: Shape = Shape(), name: String = "TensorArrayConcatV2") = run {
    buildOpTensors("TensorArrayConcatV2", name) {
      addInput(handle, false)
      addInput(flowIn, false)
      attr("dtype", dtype)
      attr("element_shape_except0", elementShapeExcept0)
    }
  }
  
  fun tensorArrayConcatV3(handle: Output, flowIn: Output, dtype: DataType<*>, elementShapeExcept0: Shape = Shape(), name: String = "TensorArrayConcatV3") = run {
    buildOpTensors("TensorArrayConcatV3", name) {
      addInput(handle, false)
      addInput(flowIn, false)
      attr("dtype", dtype)
      attr("element_shape_except0", elementShapeExcept0)
    }
  }
  
  fun tensorArrayGather(handle: Output, indices: Output, flowIn: Output, dtype: DataType<*>, elementShape: Shape = Shape(), name: String = "TensorArrayGather") = run {
    buildOpTensor("TensorArrayGather", name) {
      addInput(handle, true)
      addInput(indices, false)
      addInput(flowIn, false)
      attr("dtype", dtype)
      attr("element_shape", elementShape)
    }
  }
  
  fun tensorArrayGatherV2(handle: Output, indices: Output, flowIn: Output, dtype: DataType<*>, elementShape: Shape = Shape(), name: String = "TensorArrayGatherV2") = run {
    buildOpTensor("TensorArrayGatherV2", name) {
      addInput(handle, false)
      addInput(indices, false)
      addInput(flowIn, false)
      attr("dtype", dtype)
      attr("element_shape", elementShape)
    }
  }
  
  fun tensorArrayGatherV3(handle: Output, indices: Output, flowIn: Output, dtype: DataType<*>, elementShape: Shape = Shape(), name: String = "TensorArrayGatherV3") = run {
    buildOpTensor("TensorArrayGatherV3", name) {
      addInput(handle, false)
      addInput(indices, false)
      addInput(flowIn, false)
      attr("dtype", dtype)
      attr("element_shape", elementShape)
    }
  }
  
  fun tensorArrayGrad(handle: Output, flowIn: Output, source: String, name: String = "TensorArrayGrad") = run {
    buildOpTensor("TensorArrayGrad", name) {
      addInput(handle, false)
      addInput(flowIn, false)
      attr("source", source)
    }
  }
  
  fun tensorArrayGradV2(handle: Output, flowIn: Output, source: String, name: String = "TensorArrayGradV2") = run {
    buildOpTensor("TensorArrayGradV2", name) {
      addInput(handle, false)
      addInput(flowIn, false)
      attr("source", source)
    }
  }
  
  fun tensorArrayGradV3(handle: Output, flowIn: Output, source: String, name: String = "TensorArrayGradV3") = run {
    buildOpTensors("TensorArrayGradV3", name) {
      addInput(handle, false)
      addInput(flowIn, false)
      attr("source", source)
    }
  }
  
  fun tensorArrayGradWithShape(handle: Output, flowIn: Output, shapeToPrepend: Output, source: String, name: String = "TensorArrayGradWithShape") = run {
    buildOpTensors("TensorArrayGradWithShape", name) {
      addInput(handle, false)
      addInput(flowIn, false)
      addInput(shapeToPrepend, false)
      attr("source", source)
    }
  }
  
  fun tensorArrayPack(handle: Output, flowIn: Output, dtype: DataType<*>, elementShape: Shape = Shape(), name: String = "TensorArrayPack") = run {
    buildOpTensor("TensorArrayPack", name) {
      addInput(handle, true)
      addInput(flowIn, false)
      attr("dtype", dtype)
      attr("element_shape", elementShape)
    }
  }
  
  fun tensorArrayRead(handle: Output, index: Output, flowIn: Output, dtype: DataType<*>, name: String = "TensorArrayRead") = run {
    buildOpTensor("TensorArrayRead", name) {
      addInput(handle, true)
      addInput(index, false)
      addInput(flowIn, false)
      attr("dtype", dtype)
    }
  }
  
  fun tensorArrayReadV2(handle: Output, index: Output, flowIn: Output, dtype: DataType<*>, name: String = "TensorArrayReadV2") = run {
    buildOpTensor("TensorArrayReadV2", name) {
      addInput(handle, false)
      addInput(index, false)
      addInput(flowIn, false)
      attr("dtype", dtype)
    }
  }
  
  fun tensorArrayReadV3(handle: Output, index: Output, flowIn: Output, dtype: DataType<*>, name: String = "TensorArrayReadV3") = run {
    buildOpTensor("TensorArrayReadV3", name) {
      addInput(handle, false)
      addInput(index, false)
      addInput(flowIn, false)
      attr("dtype", dtype)
    }
  }
  
  fun tensorArrayScatter(handle: Output, indices: Output, value: Output, flowIn: Output, name: String = "TensorArrayScatter") = run {
    buildOpTensor("TensorArrayScatter", name) {
      addInput(handle, true)
      addInput(indices, false)
      addInput(value, false)
      addInput(flowIn, false)
    }
  }
  
  fun tensorArrayScatterV2(handle: Output, indices: Output, value: Output, flowIn: Output, name: String = "TensorArrayScatterV2") = run {
    buildOpTensor("TensorArrayScatterV2", name) {
      addInput(handle, false)
      addInput(indices, false)
      addInput(value, false)
      addInput(flowIn, false)
    }
  }
  
  fun tensorArrayScatterV3(handle: Output, indices: Output, value: Output, flowIn: Output, name: String = "TensorArrayScatterV3") = run {
    buildOpTensor("TensorArrayScatterV3", name) {
      addInput(handle, false)
      addInput(indices, false)
      addInput(value, false)
      addInput(flowIn, false)
    }
  }
  
  fun tensorArraySize(handle: Output, flowIn: Output, name: String = "TensorArraySize") = run {
    buildOpTensor("TensorArraySize", name) {
      addInput(handle, true)
      addInput(flowIn, false)
    }
  }
  
  fun tensorArraySizeV2(handle: Output, flowIn: Output, name: String = "TensorArraySizeV2") = run {
    buildOpTensor("TensorArraySizeV2", name) {
      addInput(handle, false)
      addInput(flowIn, false)
    }
  }
  
  fun tensorArraySizeV3(handle: Output, flowIn: Output, name: String = "TensorArraySizeV3") = run {
    buildOpTensor("TensorArraySizeV3", name) {
      addInput(handle, false)
      addInput(flowIn, false)
    }
  }
  
  fun tensorArraySplit(handle: Output, value: Output, lengths: Output, flowIn: Output, name: String = "TensorArraySplit") = run {
    buildOpTensor("TensorArraySplit", name) {
      addInput(handle, true)
      addInput(value, false)
      addInput(lengths, false)
      addInput(flowIn, false)
    }
  }
  
  fun tensorArraySplitV2(handle: Output, value: Output, lengths: Output, flowIn: Output, name: String = "TensorArraySplitV2") = run {
    buildOpTensor("TensorArraySplitV2", name) {
      addInput(handle, false)
      addInput(value, false)
      addInput(lengths, false)
      addInput(flowIn, false)
    }
  }
  
  fun tensorArraySplitV3(handle: Output, value: Output, lengths: Output, flowIn: Output, name: String = "TensorArraySplitV3") = run {
    buildOpTensor("TensorArraySplitV3", name) {
      addInput(handle, false)
      addInput(value, false)
      addInput(lengths, false)
      addInput(flowIn, false)
    }
  }
  
  fun tensorArrayUnpack(handle: Output, value: Output, flowIn: Output, name: String = "TensorArrayUnpack") = run {
    buildOpTensor("TensorArrayUnpack", name) {
      addInput(handle, true)
      addInput(value, false)
      addInput(flowIn, false)
    }
  }
  
  fun tensorArrayV2(size: Output, dtype: DataType<*>, elementShape: Shape = Shape(), dynamicSize: Boolean = false, clearAfterRead: Boolean = true, tensorArrayName: String = "", name: String = "TensorArrayV2") = run {
    buildOpTensor("TensorArrayV2", name) {
      addInput(size, false)
      attr("dtype", dtype)
      attr("element_shape", elementShape)
      attr("dynamic_size", dynamicSize)
      attr("clear_after_read", clearAfterRead)
      attr("tensor_array_name", tensorArrayName)
    }
  }
  
  fun tensorArrayV3(size: Output, dtype: DataType<*>, elementShape: Shape = Shape(), dynamicSize: Boolean = false, clearAfterRead: Boolean = true, identicalElementShapes: Boolean = false, tensorArrayName: String = "", name: String = "TensorArrayV3") = run {
    buildOpTensors("TensorArrayV3", name) {
      addInput(size, false)
      attr("dtype", dtype)
      attr("element_shape", elementShape)
      attr("dynamic_size", dynamicSize)
      attr("clear_after_read", clearAfterRead)
      attr("identical_element_shapes", identicalElementShapes)
      attr("tensor_array_name", tensorArrayName)
    }
  }
  
  fun tensorArrayWrite(handle: Output, index: Output, value: Output, flowIn: Output, name: String = "TensorArrayWrite") = run {
    buildOpTensor("TensorArrayWrite", name) {
      addInput(handle, true)
      addInput(index, false)
      addInput(value, false)
      addInput(flowIn, false)
    }
  }
  
  fun tensorArrayWriteV2(handle: Output, index: Output, value: Output, flowIn: Output, name: String = "TensorArrayWriteV2") = run {
    buildOpTensor("TensorArrayWriteV2", name) {
      addInput(handle, false)
      addInput(index, false)
      addInput(value, false)
      addInput(flowIn, false)
    }
  }
  
  fun tensorArrayWriteV3(handle: Output, index: Output, value: Output, flowIn: Output, name: String = "TensorArrayWriteV3") = run {
    buildOpTensor("TensorArrayWriteV3", name) {
      addInput(handle, false)
      addInput(index, false)
      addInput(value, false)
      addInput(flowIn, false)
    }
  }
  
  fun unstage(dtypes: Array<Long>, capacity: Long = 0L, memoryLimit: Long = 0L, container: String = "", sharedName: String = "", name: String = "Unstage") = run {
    buildOpTensors("Unstage", name) {
      attr("dtypes", dtypes)
      attr("capacity", capacity)
      attr("memory_limit", memoryLimit)
      attr("container", container)
      attr("shared_name", sharedName)
    }
  }
}