package wumo.sim.ale

import java.lang.reflect.Array.setFloat
import java.lang.reflect.Array.setInt
import org.bytedeco.javacpp.ale.ALEInterface
import wumo.sim.util.unpackDirToTemp
import java.io.InputStream
import java.lang.Exception
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import kotlin.random.Random

fun copy(source: InputStream, destination: String): Boolean {
  var success = true;
  try {
    Files.copy(source, Paths.get(destination), StandardCopyOption.REPLACE_EXISTING)
  } catch (e: Exception) {
    e.printStackTrace()
    success = false
  }
  return success
}

fun main(args: Array<String>) {
  unpackDirToTemp("atari_roms")
  
//  val source = Thread.currentThread().contextClassLoader
//      .getResourceAsStream("atari_roms/pong.bin")
//  copy(source, "/tmp/pong.bin")
  
  val ale = ALEInterface()
  
  // Get & Set the desired settings
  ale.setInt("random_seed", 123)
  //The default is already 0.25, this is just an example
  ale.setFloat("repeat_action_probability", 0.25f)
  
  ale.setBool("display_screen", true)
  ale.setBool("sound", false)
  
  // Load the ROM file. (Also resets the system for new settings to
  // take effect.)
  ale.loadROM("/tmp/atari_roms/breakout.bin")
  
  // Get the vector of legal actions
  val legal_actions = ale.legalActionSet
  
  // Play 10 episodes
  for (episode in 0..9) {
    var totalReward = 0f
    while (!ale.game_over()) {
      val a = legal_actions.get(Random.nextLong(legal_actions.limit()))
      // Apply the action and get the resulting reward
      val reward = ale.act(a).toFloat()
      totalReward += reward
    }
    println("Episode $episode ended with score: $totalReward")
    ale.reset_game()
  }
  
  System.exit(0)
}