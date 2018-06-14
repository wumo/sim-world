package wumo.sim.graphics

import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.Mesh
import com.badlogic.gdx.graphics.VertexAttribute
import com.badlogic.gdx.math.MathUtils
import com.badlogic.gdx.math.Vector2
import wumo.sim.graphics.ShapeType.Triangles
import wumo.sim.graphics.util.RawLinkedList
import java.lang.System.arraycopy

private const val sizeOfFloat = 4

enum class ShapeType(val glType: Int) {
  Point(GL20.GL_POINTS),
  Triangles(GL20.GL_TRIANGLES), Triangles_Strip(GL20.GL_TRIANGLE_STRIP), Triangles_Fan(GL20.GL_TRIANGLE_FAN),
  Lines(GL20.GL_LINES), Line_Strip(GL20.GL_LINE_STRIP), Line_Loop(GL20.GL_LINE_LOOP)
}

internal class InternalMeshPart(var primitiveType: ShapeType = Triangles,
                                var offset_in_vertex: Int = -1,
                                var max_vertices: Int = 0,
                                var mesh: Mesh? = null,
                                var node: RawLinkedList.Node<InternalMeshPart>? = null)

class MeshPartBuilder(val maxVertices: Int = 100_000) {
  private val mesh = Mesh(false, maxVertices, 0,
                          VertexAttribute.Position(),
                          VertexAttribute.ColorPacked())
  private val vertexSize_in_float = mesh.vertexAttributes.vertexSize / sizeOfFloat
  private val vertices = FloatArray(maxVertices * vertexSize_in_float)
  private var marginWriterPtrInVertex = 0
  private var freeVertexCount = maxVertices
  private var colorBits = Color(1f, 1f, 1f, 1f).toFloatBits()
  private val meshParts = RawLinkedList<InternalMeshPart>()
  private val tmp = Vector2()
  private var writerPtrInVertex = 0
  private lateinit var out_mesh: InternalMeshPart
  private val tmpColor = Color()
  private var modify = false
  
  init {
    mesh.setVertices(vertices)
  }
  
  private fun write1vertex(x: Float, y: Float) {
    var idx = writerPtrInVertex * vertexSize_in_float
    vertices[idx++] = x
    vertices[idx++] = y
    vertices[idx++] = 0f
    vertices[idx] = colorBits
    writerPtrInVertex++
  }
  
  fun vertex(x: Float, y: Float) {
    claimVertices(1)
    write1vertex(x, y)
  }
  
  fun color(colorBits: Float) {
    this.colorBits = colorBits
  }
  
  fun color(color: Color) {
    this.colorBits = color.toFloatBits()
  }
  
  fun color(r: Float, g: Float, b: Float, a: Float) {
    tmpColor.set(r, g, b, a)
    this.colorBits = tmpColor.toFloatBits()
  }
  
  fun rectLine(x1: Float, y1: Float, x2: Float, y2: Float, width: Float) {
    var width = width
    val t = tmp.set(y2 - y1, x1 - x2).nor()
    width *= 0.5f
    val tx = t.x * width
    val ty = t.y * width
    vertex(x1 + tx, y1 + ty)
    vertex(x1 - tx, y1 - ty)
    vertex(x2 + tx, y2 + ty)
    
    vertex(x2 - tx, y2 - ty)
    vertex(x2 + tx, y2 + ty)
    vertex(x1 - tx, y1 - ty)
  }
  
  fun line(x1: Float, y1: Float, x2: Float, y2: Float) {
    claimVertices(2)
    write1vertex(x1, y1)
    write1vertex(x2, y2)
  }
  
  fun circle(x: Float, y: Float, radius: Float, segments: Int = Math.max(1, (6 * Math.cbrt(radius.toDouble())).toInt())) {
    if (segments <= 0) throw IllegalArgumentException("segments must be > 0.")
    val angle = 2 * MathUtils.PI / segments
    val cos = MathUtils.cos(angle)
    val sin = MathUtils.sin(angle)
    var cx = radius
    var cy = 0f
    if (out_mesh.primitiveType == ShapeType.Lines) {
      for (i in 0 until segments) {
        vertex(x + cx, y + cy)
        val temp = cx
        cx = cos * cx - sin * cy
        cy = sin * temp + cos * cy
        vertex(x + cx, y + cy)
      }
      // Ensure the last segment is identical to the first.
      vertex(x + cx, y + cy)
    } else {
      for (i in 0 until segments - 1) {
        vertex(x, y)
        vertex(x + cx, y + cy)
        val temp = cx
        cx = cos * cx - sin * cy
        cy = sin * temp + cos * cy
        vertex(x + cx, y + cy)
      }
      // Ensure the last segment is identical to the first.
      vertex(x, y)
      vertex(x + cx, y + cy)
    }
    
    cx = radius
    cy = 0f
    vertex(x + cx, y + cy)
  }
  
  fun triangle_filled(x1: Float, y1: Float, x2: Float, y2: Float, x3: Float, y3: Float) {
    claimVertices(3)
    write1vertex(x1, y1)
    write1vertex(x2, y2)
    write1vertex(x3, y3)
  }
  
  fun rect_filled(x00: Float, y00: Float, x10: Float, y10: Float, x11: Float, y11: Float,
                  x01: Float, y01: Float) {
    triangle_filled(x00, y00, x10, y10, x11, y11)
    triangle_filled(x11, y11, x01, y01, x00, y00)
  }
  
  internal fun begin(meshPart: InternalMeshPart, type: ShapeType = Triangles) {
    out_mesh = meshPart
    out_mesh.primitiveType = type
    out_mesh.mesh = mesh
    if (out_mesh.node == null) {
      modify = false
      out_mesh.offset_in_vertex = marginWriterPtrInVertex
      out_mesh.node = meshParts.linkLast(out_mesh)
    } else
      modify = true
    writerPtrInVertex = out_mesh.offset_in_vertex
  }
  
  internal fun end() {
    val writtenVertices = writerPtrInVertex - out_mesh.offset_in_vertex
    val extraFreeVertices = out_mesh.max_vertices - writtenVertices
    freeVertexCount += extraFreeVertices
    out_mesh.max_vertices = writtenVertices
    mesh.updateVertices(out_mesh.offset_in_vertex * vertexSize_in_float,
                        vertices, out_mesh.offset_in_vertex * vertexSize_in_float,
                        out_mesh.max_vertices * vertexSize_in_float)
  }
  
  internal fun remove(a: InternalMeshPart) {
    meshParts.unlink(a.node!!)
    freeVertexCount += a.max_vertices
    a.offset_in_vertex = -1
    a.max_vertices = 0
    a.node = null
  }
  
  private fun compact() {
    var offset_in_vertex = 0
    for (mp in meshParts) {//可能包含未确定最大顶点数的allocation（动态allocation）
      mp!!
      align(mp, offset_in_vertex)
      offset_in_vertex += mp.max_vertices
    }
    val delta_in_vertex = marginWriterPtrInVertex - offset_in_vertex
    writerPtrInVertex -= delta_in_vertex
    marginWriterPtrInVertex -= delta_in_vertex
  }
  
  private fun align(meshPart: InternalMeshPart, offset_in_vertex: Int) {
    //动态allocation的偏置是有意义的，可以移位处理。
    if (meshPart.offset_in_vertex == offset_in_vertex) return
    //动态allocation的max_vertices==0，所以不用移顶点数据。
    if (meshPart.max_vertices > 0) {
      System.arraycopy(vertices, meshPart.offset_in_vertex * vertexSize_in_float,
                       vertices, offset_in_vertex * vertexSize_in_float,
                       meshPart.max_vertices * vertexSize_in_float)
      mesh.updateVertices(offset_in_vertex * vertexSize_in_float,
                          vertices, offset_in_vertex * vertexSize_in_float,
                          meshPart.max_vertices * vertexSize_in_float)
    }
    meshPart.offset_in_vertex = offset_in_vertex
  }
  
  private fun claimVertices(numVertices: Int) {
    if (numVertices <= 0) return
    if (modify) {//modify existing allocation
      val writtenVertices = writerPtrInVertex - out_mesh.offset_in_vertex
      val totalVertices = writtenVertices + numVertices
      if (totalVertices > out_mesh.max_vertices) {//exceeding previous allocation, make a larger allocation at the end
        val extraVertices = totalVertices - out_mesh.max_vertices
        if (freeVertexCount < extraVertices)
          throw Exception("not enough memory for more vertices!!$freeVertexCount<$numVertices @$marginWriterPtrInVertex")
        if (out_mesh.node !== meshParts.last || marginWriterPtrInVertex + extraVertices > maxVertices) {
          if (marginWriterPtrInVertex + totalVertices > maxVertices) {
            val backup = FloatArray(writtenVertices * vertexSize_in_float)
            arraycopy(vertices, out_mesh.offset_in_vertex * vertexSize_in_float,
                      backup, 0,
                      backup.size)//backup already written vertices
            meshParts.unlink(out_mesh.node!!)
            compact()
            arraycopy(backup, 0,
                      vertices, marginWriterPtrInVertex * vertexSize_in_float,
                      backup.size)
          } else {
            //allocate at the end
            arraycopy(vertices, out_mesh.offset_in_vertex * vertexSize_in_float,
                      vertices, marginWriterPtrInVertex * vertexSize_in_float,
                      writtenVertices * vertexSize_in_float)
            meshParts.unlink(out_mesh.node!!)
          }
          out_mesh.offset_in_vertex = marginWriterPtrInVertex
          out_mesh.node = meshParts.linkLast(out_mesh)
          
          marginWriterPtrInVertex += writtenVertices
          writerPtrInVertex = marginWriterPtrInVertex
        }//last allocation and there are still extra vertices
        out_mesh.max_vertices += extraVertices
        freeVertexCount -= extraVertices
        marginWriterPtrInVertex += extraVertices
      }
    } else {//new allocation
      if (freeVertexCount < numVertices)
        throw Exception("not enough memory for more vertices!!$freeVertexCount<$numVertices @$marginWriterPtrInVertex")
      if (marginWriterPtrInVertex + numVertices > maxVertices)
        compact()
      freeVertexCount -= numVertices
      marginWriterPtrInVertex += numVertices
      out_mesh.max_vertices += numVertices
    }
  }
  
  fun dispose() {
    mesh.dispose()
  }
}
