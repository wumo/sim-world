package wumo.sim.graphics

import com.badlogic.gdx.graphics.glutils.ShaderProgram
import com.badlogic.gdx.graphics.glutils.ShaderProgram.COLOR_ATTRIBUTE
import com.badlogic.gdx.graphics.glutils.ShaderProgram.POSITION_ATTRIBUTE
import com.badlogic.gdx.math.Matrix4

class MeshPartRender {
  val uProjModelView = "u_ProjModelView"
  val transformMatrix = Matrix4()
  val projectionMatrix = Matrix4()
  private val combinedMatrix = Matrix4()
  private var shader = createDefaultShader()
  
  internal fun render(meshPart: InternalMeshPart) {
    combinedMatrix.set(projectionMatrix)
    Matrix4.mul(combinedMatrix.`val`, transformMatrix.`val`)
    
    shader.setUniformMatrix(uProjModelView, combinedMatrix)
    meshPart.apply {
      mesh!!.render(shader, primitiveType.glType, offset_in_vertex, max_vertices)
    }
  }
  
  fun dispose() {
    shader.dispose()
  }
  
  fun createDefaultShader(): ShaderProgram {
    val vertexShader = """
      attribute vec4 $POSITION_ATTRIBUTE;
      attribute vec4 $COLOR_ATTRIBUTE;
      uniform mat4 $uProjModelView;
      varying vec4 v_col;
      void main() {
        gl_Position = $uProjModelView * $POSITION_ATTRIBUTE;
        v_col = $COLOR_ATTRIBUTE;
        gl_PointSize = 1.0;
      }
      """
    val fragmentShader = """
      #ifdef GL_ES
        precision mediump float;
      #endif
      varying vec4 v_col;
      void main() {
        gl_FragColor = v_col;
      }
      """
    return ShaderProgram(vertexShader, fragmentShader)
  }
  
  fun begin() {
    shader.begin()
  }
  
  fun end() {
    shader.end()
  }
}
