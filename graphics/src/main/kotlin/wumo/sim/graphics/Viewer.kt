@file:Suppress("NOTHING_TO_INLINE")

package wumo.sim.graphics

import com.badlogic.gdx.ApplicationListener
import com.badlogic.gdx.Gdx
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3Application
import com.badlogic.gdx.backends.lwjgl3.Lwjgl3ApplicationConfiguration
import com.badlogic.gdx.graphics.GL20
import com.badlogic.gdx.graphics.OrthographicCamera
import com.badlogic.gdx.graphics.g2d.SpriteBatch
import com.badlogic.gdx.math.Quaternion
import com.badlogic.gdx.math.Vector3
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CountDownLatch
import kotlin.concurrent.thread

class Config(val width: Int, val height: Int,
             val maxVertices: Int = 100_000,
             val isContinousRendering: Boolean = true)

class Viewer(val config: Config) : ApplicationListener {
  private val geoms = ConcurrentHashMap.newKeySet<Geom>()
  private val geomsToRemove = ConcurrentHashMap.newKeySet<Geom>()
  private val imgs = ConcurrentHashMap.newKeySet<Image>()
  lateinit var camera: OrthographicCamera
  lateinit var builder: MeshPartBuilder
  lateinit var renderer: MeshPartRender
  lateinit var sprite: SpriteBatch
  private var running = false
  private var startCallback = {}
  private var closeCallback = {}
  
  fun start() {
    val app_config = Lwjgl3ApplicationConfiguration()
    app_config.apply {
      useOpenGL3(true, 3, 3)
      setWindowedMode(config.width, config.height)
      setWindowPosition(0, 30)
      setBackBufferConfig(8, 8, 8, 8, 32, 0, 8)
    }
    running = true
    Lwjgl3Application(this, app_config)
    closeCallback()
    running = false
  }
  
  fun startAsync() {
    val latch = CountDownLatch(1)
    startCallback = { latch.countDown() }
    thread {
      start()
    }
    latch.await()
  }
  
  fun close() {
    if (!running) return
    val latch = CountDownLatch(1)
    closeCallback = { latch.countDown() }
    Gdx.app.exit()
    latch.await()
  }
  
  inline operator fun plusAssign(geom: Geom) = add(geom)
  inline operator fun minusAssign(geom: Geom) = remove(geom)
  inline operator fun plusAssign(img: Image) = add(img)
  inline operator fun minusAssign(img: Image) = remove(img)
  
  fun add(geom: Geom) {
    geoms.add(geom)
    geom.changed = true
  }
  
  fun remove(geom: Geom) {
    geomsToRemove.add(geom)
  }
  
  fun add(img: Image) {
    imgs += img
  }
  
  fun remove(img: Image) {
  }
  
  fun requestRender() {
    Gdx.graphics.requestRendering()
  }
  
  override fun create() {
    camera = OrthographicCamera(Gdx.graphics.width.toFloat(), Gdx.graphics.height.toFloat())
    builder = MeshPartBuilder(config.maxVertices)
    renderer = MeshPartRender()
    sprite = SpriteBatch()
    Gdx.graphics.isContinuousRendering = config.isContinousRendering
    startCallback()
  }
  
  override fun render() {
    Gdx.gl.glClearColor(0.417f, 0.417f, 0.417f, 0f)
    Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT or GL20.GL_DEPTH_BUFFER_BIT)
    Gdx.gl.glEnable(GL20.GL_DEPTH_TEST)
    camera.update()
    renderer.projectionMatrix.set(camera.combined)
    renderer.begin()
    val iter = geomsToRemove.iterator()
    while (iter.hasNext()) {
      builder.remove(iter.next().mesh)
      iter.remove()
    }
    val tmp_translation = Vector3()
    val tmp_rotation = Quaternion()
    val tmp_scale = Vector3()
    for (geom in geoms) {
      geom.prepareMesh(builder)
      geom.attr(geom)
      tmp_translation.set(geom.translation, geom.z)
      tmp_rotation.setFromAxisRad(0f, 0f, 1f, geom.rotation)
      tmp_scale.set(geom.scale, 1f)
      renderer.transformMatrix.set(tmp_translation, tmp_rotation, tmp_scale)
      renderer.render(geom.mesh)
    }
    renderer.end()
    
    sprite.begin()
    for (img in imgs) {
      img.prepare()
      sprite.draw(img.tex, 0f, 0f)
    }
    sprite.end()
  }
  
  override fun pause() {
  }
  
  override fun resume() {
  }
  
  override fun resize(width: Int, height: Int) {
    camera.setToOrtho(false, width.toFloat(), height.toFloat())
  }
  
  override fun dispose() {
    renderer.dispose()
    builder.dispose()
    for (img in imgs)
      img.dispose()
  }
}
