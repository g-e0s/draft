package scalars.lib.recsys

abstract class RecSys {
  def recommend(user: String, k: Int): List[String] = ???
}