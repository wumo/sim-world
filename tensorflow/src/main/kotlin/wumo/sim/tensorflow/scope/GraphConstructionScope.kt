package wumo.sim.tensorflow.scope

import org.bytedeco.javacpp.tensorflow
import wumo.sim.tensorflow.core.Graph
import wumo.sim.tensorflow.ops.Op
import wumo.sim.tensorflow.ops.OpSpecification
import wumo.sim.tensorflow.ops.control_flow_ops.ControlFlowContext

class GraphConstructionScope(
    val graph: Graph = Graph(),
    var nameScope: NameScope = NameScope(""),
    var device: String = "",
    var deviceFunction: (OpSpecification) -> String = { it.device },
    val colocationOps: MutableSet<Op> = mutableSetOf(),
    val controlDependencies: MutableSet<Op> = mutableSetOf(),
    val attributes: MutableMap<String, tensorflow.AttrValue> = mutableMapOf(),
    var container: String = "", // TODO: !!! Use containers.
    var controlFlowContext: ControlFlowContext? = null,
    var outerContext: GraphConstructionScope? = null)