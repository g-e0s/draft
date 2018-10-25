package scalars.lib.offer

import net.liftweb.json.JsonAST.JObject
import net.liftweb.json.JsonDSL._
import net.liftweb.json._

trait Condition {
  val condition: JObject
}

class OrderSumCondition(sumScale: List[Long], discountScale: List[Double]) extends Condition {
  override val condition: JObject = _
}