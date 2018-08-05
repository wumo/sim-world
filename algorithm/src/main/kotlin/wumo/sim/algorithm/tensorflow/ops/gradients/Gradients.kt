package wumo.sim.algorithm.tensorflow.ops.gradients

import org.bytedeco.javacpp.tensorflow.*
import wumo.sim.algorithm.tensorflow.Op
import wumo.sim.algorithm.tensorflow.TF
import wumo.sim.algorithm.tensorflow.Tensor
import wumo.sim.algorithm.tensorflow.ops.gen.addN
import wumo.sim.algorithm.tensorflow.ops.register_array_grad
import wumo.sim.algorithm.tensorflow.ops.register_math_grad
import wumo.sim.algorithm.tensorflow.ops.register_nn_grad
import wumo.sim.algorithm.tensorflow.tf
import java.util.*
import kotlin.collections.HashMap
import kotlin.collections.HashSet
import kotlin.collections.Iterator
import kotlin.collections.List
import kotlin.collections.MutableCollection
import kotlin.collections.MutableList
import kotlin.collections.contains
import kotlin.collections.isNotEmpty
import kotlin.collections.iterator
import kotlin.collections.mutableListOf
import kotlin.collections.set
import kotlin.collections.toTypedArray
import wumo.sim.algorithm.tensorflow.ops.gen.onesLike as _onesLike
import wumo.sim.algorithm.tensorflow.ops.gen.zerosLike as _zerosLike

fun TF.addSymbolicGradients(outputs: List<Tensor>,
                            inputs: List<Tensor>): MutableList<Tensor> {
  val grad_inputs = MutableList(outputs.size) {
    _onesLike(outputs[it])
  }
  return addSymbolicGradients(outputs, inputs, grad_inputs)
}

fun TF.addSymbolicGradients(outputs: List<Tensor>,
                            inputs: List<Tensor>,
                            grad_inputs: List<Tensor>): MutableList<Tensor> {
  val out = MutableList(inputs.size) { noGradient }
  val builder = SymbolicGradientBuilder(this, outputs, inputs, grad_inputs, out)
  builder.addGradients()
  return out
}

/**A vector of output endpoints which represents backpropagated gradients.*/
typealias BackproppedGradients = ArrayList<Tensor>

typealias GradFunc = (Op, List<Tensor>, MutableList<Tensor>) -> Unit

val nullGradFunc: GradFunc = { _, _, _ -> }

object GradOpRegistry {
  val map = HashMap<String, GradFunc>()
  
  init {
    register_math_grad()
    register_nn_grad()
    register_array_grad()
  }
  
  operator fun set(name: String, fn: GradFunc) {
    map[name] = fn
  }
  
  operator fun get(name: String): GradFunc {
    if (name !in map)
      throw NotImplementedError("No gradient defined for op: $name")
    return map[name]!!
  }
}

fun register_gradient_op(vararg names: String, fn: GradFunc) {
  for (name in names)
    register_gradient_op_uniq(name, fn)
}

fun register_no_gradient_op(vararg names: String) {
  for (name in names)
    register_gradient_op_uniq(name, nullGradFunc)
}

fun register_gradient_op_uniq(name: String, fn: GradFunc) {
  GradOpRegistry[name] = fn
}

class SymbolicGradientBuilder(val tf: TF,
                              val outputs_: List<Tensor>,
                              val inputs_: List<Tensor>,
                              val grad_inputs_: List<Tensor>,
                              val grad_outputs_: MutableList<Tensor>) {
  /**
   * backprops_ is a map from a node output to its accumulated
   * gradients.  When a node output has accumulated all its
   * gradients, we add a node which sums them up.
   */
  val backprops_ = HashMap<Tensor, BackproppedGradients>()
  /**
   * pending[i] is count-down counter for i-th node's expected
   * backprops.  When pending[i] becomes zero, we collected all
   * backprop gradients for all outputs of the ith-node.
   */
  lateinit var pending_: MutableList<Int>
  /**
   * `ready` keeps track of nodes that have been completely
   * backpropped. Initially, for every output in `outputs_`, we add initial
   * gradients from `grad_inputs_`.
   */
  val ready_ = ArrayDeque<Node>()
  /**
   * The set of node ids in `inputs_`. Used to identify nodes at backprop
   * frontier. Maps from Output -> index into `grad_outputs_`.
   */
  val input_nodes_ = HashMap<Tensor, Int>()
  
  private fun initialize() {
    if (outputs_.size != grad_inputs_.size)
      throw IllegalArgumentException("Must specify a gradient input for each output.")
    val reachable_nodes = getReachableNodes()
    for (input in inputs_)
      if (!reachable_nodes[input.node().id()])
        throw IllegalArgumentException("Cannot compute the partial derivative for node " +
                                           "'${input.node().name().string}'" +
                                           " as it's unreachable from the output node(s).")
//    grad_outputs_.clear()
    
    val output_nodes = HashSet<Int>()
    for (i in 0 until outputs_.size)
      output_nodes.add(outputs_[i].node().id())
    
    val stop_backprop_nodes = getStopBackpropNodes(reachable_nodes, output_nodes)
    
    // Populate `input_nodes_` from Outputs in `inputs_`.
    for (i in 0 until inputs_.size)
      input_nodes_[inputs_[i]] = i
    
    // TODO(andydavis) Consider a more efficient data structure for `pending_` to
    // handle computing gradients over small subgraphs from a very large graph.
    pending_ = MutableList(tf.g.num_node_ids()) { 0 }
    backprops_.clear()
    val visited = HashSet<Node>()
    val queue = ArrayDeque<Node>()
    for (nout in inputs_) {
      val n = nout.node()
      if (n !in visited) {
        queue.addLast(n)
        visited.add(n)
      }
    }
    
    // Going forward to figure out which endpoints need backprop-ed.
    // A node's endpoints need to be backprop-ed only if one of the
    // arg node can reach the node via data edges.
    while (queue.isNotEmpty()) {
      val n = queue.removeFirst()
      for (i in 0 until n.num_outputs())
        backprops_[toTensor(n, i)] = BackproppedGradients()
      var num_expected_backprops = 0
      if (n.id() !in stop_backprop_nodes) {
        // Internal node: continue BFS along connected outputs.
        for (e in n.out_edges().iterate()) {
          val dst = e.dst()
          // If a node is not reachable from outputs_,
          // we don't expect it to receive a backpropagated gradient.
          // It will not be counted in num_expected_backprops.
          if (e.IsControlEdge() || !reachable_nodes[dst.id()]) continue
          if (dst !in visited) {
            queue.addLast(dst)
            visited.add(dst)
          }
          ++num_expected_backprops
        }
      }
      if (n.id() in output_nodes) {
        // Output node: update `num_expected_backprops` for each Output in
        // `outputs_` that references `n`.
        for (output in outputs_)
          if (output.node() == n)
            ++num_expected_backprops
      }
      pending_[n.id()] = num_expected_backprops
    }
    
    // Initialize backprop with `grad_inputs_`.
    val num_dy = grad_inputs_.size
    for (i in 0 until num_dy)
      backpropAlongEdge(grad_inputs_[i], outputs_[i])
  }
  
  /**
   * For each forward edge from `src` to `dst` in the initial/forward graph:
   * propagates gradients `dst_grad` backwards along the edge from `src`
   * to `dst` in the graph. This will add `dst_grad` to the list of pending
   * gradients for the node associated with `src`.
   */
  private fun backpropAlongEdge(dst_grad: Tensor, src: Tensor) {
    if (src.node() == null)
      throw Exception("Attempted to backprop along an invalid edge.")
    val grads = backprops_[src]
    if (grads != null) {
      grads.add(dst_grad)
      if (--pending_[src.node().id()] == 0)
        ready_.addLast(src.node())
    }
  }
  
  /**
   * Gets the set of node ids at which to stop backprop. These are all elements
   * of `outputs_` that do not get transitively consumed by other `outputs_`.
   * Used to identify nodes at which to stop backprop.
   */
  private fun getStopBackpropNodes(reachable_nodes: List<Boolean>,
                                   output_nodes: HashSet<Int>): HashSet<Int> {
    // Output nodes that get transitively consumed by other `outputs_` are stored
    // in `internal_outputs`.
    val internal_outputs = HashSet<Int>()
    val visited = HashSet<Node>()
    // Initialize `queue` for BFS traversal. Nodes in `queue` hold upcoming nodes
    // along with the last Node in `output_` encountered along that path. If no
    // `output_` node was encountered, pair.second will be nullptr.
    val queue = ArrayDeque<Pair<Node, Node?>>()
    for (nout in inputs_) {
      val n = nout.node()
      if (n !in visited) {
        queue.addLast(Pair(n, null))
        visited.add(n)
      }
    }
    // BFS from nodes in 'inputs_' along out edges for the entire graph. Internal
    // output nodes are recorded during the traversal. All nodes that are output
    // nodes but not internal output nodes are considered the frontier of the
    // output nodes, and thus our stop backprop nodes.
    while (queue.isNotEmpty()) {
      val p = queue.removeFirst()
      val n = p.first
      for (e in n.out_edges().iterate()) {
        // If a node is not reachable from outputs_, we can stop.
        if (e.IsControlEdge() || !reachable_nodes[e.dst().id()]) continue
        if (e.dst() in visited) continue//TODO 这里可能是错的，导致内部output_node被错误识别为stop_backprop_node
        
        val dst = e.dst()
        var last_output_node = p.second
        if (dst.id() in output_nodes) {
          // We reached an output node.
          if (last_output_node != null) {
            // If we had already found an output node on this path so we mark
            // it as an internal output.
            internal_outputs.add(last_output_node.id())
          }
          // Mark this newly found output node to insert in the queue.
          last_output_node = dst
        }
        queue.addLast(Pair(dst, last_output_node))
        visited.add(dst)
      }
    }
    val stop_backprop_nodes = HashSet<Int>()
    for (output_node in output_nodes)
      if (output_node !in internal_outputs)
        stop_backprop_nodes.add(output_node)
    return stop_backprop_nodes
  }
  
  /**
   * Returns a list mapping whether each node in the graph is reachable
   * from outputs_. Keyed by node id.
   */
  private fun getReachableNodes(): List<Boolean> {
    val num_node_ids = tf.g.c_graph.graph().num_node_ids()
    val reachable_nodes = MutableList(num_node_ids) { false }
    val queue = ArrayDeque<Node>()
    val visited = MutableList(num_node_ids) { false }
    for (out in outputs_) {
      val n = out.node()
      if (!reachable_nodes[n.id()]) {
        queue.addLast(n)
        reachable_nodes[n.id()] = true
      }
    }
    
    //BFS
    while (queue.isNotEmpty()) {
      val n = queue.removeFirst()
      for (e in n.in_edges().iterate()) {
        if (e.IsControlEdge()) continue
        val src = e.src()
        if (visited[src.id()]) continue
        queue.addLast(src)
        reachable_nodes[src.id()] = true
        visited[src.id()] = true
      }
    }
    return reachable_nodes
  }
  
  fun addGradients() {
    // Initialize backprops.
    initialize()
    
    // Backward propagation.
    
    while (ready_.isNotEmpty()) {
      // n has collected all gradients.
      val n = ready_.removeFirst()
      
      // dy[i] is the sum of i-th output's backpropped gradients.
      val num_y = n.num_outputs()
      val dy = MutableList(num_y) { toTensor(null, 0) }
      val no_grad_dy_indices = mutableListOf<Int>()
      for (i in 0 until num_y) {
        dy[i] = sumGradients(toTensor(n, i))
        if (dy[i] == noGradient)
          no_grad_dy_indices.add(i)
        val id = input_nodes_[toTensor(n, i)]
        if (id != null) {
          // Return gradients for Output in 'grad_outputs_'.
          grad_outputs_[id] = dy[i]
        }
      }
      
      // Stop backprop if none of the inputs to `n` are in `backprops_'.
      var stop_node = true
      for (e in n.in_edges().iterate()) {
        if (e.IsControlEdge()) continue
        if (backprops_.contains(toTensor(e.src(), e.src_output()))) {
          stop_node = false
          break
        }
      }
      if (stop_node) continue
      
      // Special case: if we find an exit node, process the associated while loop.
      // Note that ProcessWhileLoop() calls BackpropAlongEdge() if necessary
      // (which updates ready_), and we skip all the regular processing below
      // after calling it.
      if (n.IsExit()) {
        assert(dy.size == 1)
        //TODO ProcessWhileLoop(n,dy[0])
        continue
      }
      // All loop-specific control flow ops should have been handled above
      assert(!n.IsEnter() && !n.IsNextIteration()) { n.DebugString().string }
      
      val num_no_grad = no_grad_dy_indices.size
      if (isPrimitiveOpWithNoGrad(n.type_string().string) || num_no_grad == num_y) {
        // No grad defined for this op, or all outputs returned 'NoGradient':
        // Backprop 'NoGradient' along the in edges.
        for (e in n.in_edges().iterate()) {
          if (e.IsControlEdge()) continue
          backpropAlongEdge(noGradient, toTensor(e.src(), e.src_output()))
        }
        continue
      }
      
      if (num_no_grad in 1..(num_y - 1)) {
        // The outputs of 'n' returned a mixture of valid gradients and
        // 'NoGradient'. Therefore, we need to add 'ZerosLike' nodes for each
        // 'NoGradient' output before we call the gradient function for 'n'.
        // TODO(andydavis) If static shapes are known, replace 'ZerosLike' with
        // zero-filled Constant node of appropriate shape.
        
        for (dy_index in no_grad_dy_indices)
          dy[dy_index] = tf._zerosLike(toTensor(n, dy_index))
      }
      
      // TODO(andydavis) Add option to encapsulate grad function in
      // SymbolicGradientOp (as opposed to inlining into the graph).
      val dx = mutableListOf<Tensor>()
      callGradFunction(Operation(n), dy, dx)
      
      // Backprop along the in edges.
      // TODO(andydavis) Find cleaner way to map each grad output returned by
      // gradient function to the src node/output to which it should be
      // backproped. Maybe grad functions can return a vector of Output pairs to
      // make this association explicit.
      var dx_index = 0
      for (e in n.in_edges().iterate()) {
        if (e.IsControlEdge()) continue
        if (dx_index == dx.size)
          throw Exception("Invalid gradient output index: $dx_index size: ${dx.size}")
        backpropAlongEdge(dx[dx_index++], toTensor(e.src(), e.src_output()))
      }
    }
    
    // Check if any input nodes still have pending gradients and have not been
    // processed yet. This happens if not all outputs of a node are in 'inputs_'.
    val requested_grads = HashMap<Node, Int>()
    for (nout in inputs_) {
      if (pending_[nout.node().id()] > 0) {
        assert(nout.node().num_outputs() > 1)
        val idx = input_nodes_[nout]!!
        assert(grad_outputs_[idx].node().isNull)
        grad_outputs_[idx] = sumGradients(nout)
        requested_grads.compute(nout.node()) { _, v -> (v ?: 0) + 1 }
      }
    }
    for (p in requested_grads.entries) {
      val num_requested_inputs = p.key.num_outputs() - pending_[p.key.id()]
      assert(num_requested_inputs == p.value)
    }
  }
  
  private fun callGradFunction(op: Operation, grad_inputs: MutableList<Tensor>, grad_outputs: MutableList<Tensor>) {
    val grad_fn = GradOpRegistry[op.node().type_string().string]!!
    grad_fn(op.node().toOperation(),
            grad_inputs, grad_outputs)
  }
  
  private fun isPrimitiveOpWithNoGrad(opname: String): Boolean {
    val grad_fn = GradOpRegistry[opname]
    return grad_fn == nullGradFunc
  }
  
  /**
   * Adds a node to the graph (returned in `grad`) that sums the in-bound
   * gradients to `src` (if there are more than one).
   */
  private fun sumGradients(src: Tensor): Tensor {
    val grads = backprops_[src]
        ?: throw  Exception("Unable to find backprop list for node.id ${src.node().name().string}")
    
    // Filter any backproped 'NoGradient' Outputs from 'grads' (if needed).
    // Return any valid backproped gradients that remain after filtering,
    // or 'NoGradient' otherwise.
    val grads_to_keep = mutableListOf<Tensor>()
    for (o in grads) {
      if (o == noGradient) continue
      grads_to_keep.add(o)
    }
    
    return when {
      grads_to_keep.isEmpty() ->
        // Nothing propagated back. Return 'NoGradient'.
        noGradient
      grads_to_keep.size == 1 ->
        // Just one backprop edge.
        grads_to_keep[0]
      else -> {
        // Otherwise, adds backprop-ed gradients.
        // TODO(andydavis) Use a better accumulator here.
        tf.addN(grads_to_keep.toTypedArray())
      }
    }
  }
}

val noGradient = toTensor(null, -1)
fun toTensor(n: Node?, v: Int) =
    if (n == null) Tensor(null, v)
    else
      Tensor(n.toOperation(), v)

fun Node.toOperation() = Op(tf.g, TF_Operation(this))

internal fun EdgeSet.iterate() = object : Iterator<Edge> {
  val iter = EdgeSetIterator(begin())
  val end = EdgeSetIterator(end())
  override fun hasNext() = iter.notEquals(end)
  
  override fun next(): Edge {
    val e = Edge(iter.access().get())
    iter.increment()
    return e
  }
}

inline fun <E> MutableCollection<E>.append(vararg elements: E) {
  for (element in elements)
    add(element)
}