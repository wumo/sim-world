package wumo.sim.envs.atari

import org.bytedeco.javacpp.IntPointer
import org.bytedeco.javacpp.ale.ALEInterface
import wumo.sim.core.Env
import wumo.sim.core.Space
import wumo.sim.envs.atari.AtariEnv.Companion.ObsType.image
import wumo.sim.envs.atari.AtariEnv.Companion.ObsType.ram
import wumo.sim.graphics.Config
import wumo.sim.graphics.Viewer
import wumo.sim.spaces.Box
import wumo.sim.spaces.Discrete
import wumo.sim.util.*
import wumo.sim.util.ndarray.NDArray
import wumo.sim.utils.hash_seed
import wumo.sim.utils.np_random
import java.io.File.separatorChar
import kotlin.random.Random

class AtariEnv(val game: String = "pong",
               val obs_type: ObsType = ram,
               val frameskip: Pair<Int, Int> = 2 to 5,
               repeat_action_probability: Float = 0f) : Env<NDArray<Byte>, Int> {
  
  companion object {
    val atari_roms_path = "atari_roms"
    val game_dir: String = unpackDirToTemp(atari_roms_path)
    val games: Set<String> = listResources(atari_roms_path)
        .map { it.nameWithoutExtension }
        .toSet()
    
    enum class ObsType {
      ram, image
    }
  }
  
  lateinit var rand: Random
  val game_path = "$game_dir$separatorChar$game.bin"
  val ale = ALEInterface()
  val action_set: IntPointer
  val screen_width: Int
  val screen_height: Int
  override val action_space: Space<Int>
  override val observation_space: Space<NDArray<Byte>>
  
  init {
    require(game in games)
    ale.setFloat("repeat_action_probability", repeat_action_probability)
    
    seed()
    
    action_set = ale.minimalActionSet
    action_space = Discrete(action_set.limit().toInt())
    
    val screen = ale.screen
    screen_width = screen.width().toInt()
    screen_height = screen.height().toInt()
    
    observation_space = when (obs_type) {
      ram -> Box(0.toByte(), 255.toByte(), Shape(128))
      image -> Box(0.toByte(), 255.toByte(), Shape(screen_height, screen_width, 3))
    }
  }
  
  override fun seed(seed: Long?): List<Long> {
    val (rand, seed1) = np_random(seed)
    val seed2 = hash_seed(seed1 + 1) % (1 shl 31)
    ale.setInt("random_seed", seed2.toInt())
    ale.loadROM(game_path)
    return listOf(seed1, seed2)
  }
  
  override fun step(a: Int): t4<NDArray<Byte>, Float, Boolean, Map<String, Any>> {
    var reward = 0f
    val action = action_set[a.toLong()]
    val num_steps = rand.nextInt(frameskip._1, frameskip._2)
    repeat(num_steps) {
      reward += ale.act(action)
    }
    val ob = get_obs()
    
    return t4(ob, reward, ale.game_over(), mapOf("ale.lives" to ale.lives()))
  }
  
  private fun get_obs(): NDArray<Byte> =
      when (obs_type) {
        ram -> getRam()
        image -> getImage()
      }
  
  private fun getRam(): NDArray<Byte> {
    val ram = ale.ram.array()
    val array = ByteArray(ram.limit().toInt())
    ram.get(array)
    return NDArray(array)
  }
  
  private fun getImage(): NDArray<Byte> {
    val screen_size = screen_width * screen_height
    val array = ByteArray(screen_size)
    ale.getScreenRGB(array)
    return NDArray(Shape(screen_height, screen_width, 3), array)
  }
  
  override fun reset(): NDArray<Byte> {
    ale.reset_game()
    return get_obs()
  }
  
  lateinit var viewer: Viewer
  override fun render() {
    val img = getImage()
    if (!::viewer.isInitialized) {
      viewer = Viewer(Config(screen_width, screen_height, isContinousRendering = false))
      
    }
    viewer.requestRender()
    Thread.sleep(1000 / 60)
  }
  
  override fun close() {
    if (::viewer.isInitialized)
      viewer.close()
  }
  
}