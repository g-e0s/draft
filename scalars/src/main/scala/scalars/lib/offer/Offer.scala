package scalars.lib.offer

import net.liftweb.json.JsonAST.JObject
import net.liftweb.json.JsonDSL._
import net.liftweb.json._


sealed trait Offer {
  val recs: JObject = Nil
  def format(): String = JsonAST.prettyRender(this.recs)
}

case class ItemsOffer(items: List[ItemDiscount]) extends Offer {
  val reward: List[JObject] = items.map(item =>
      ("type" -> item.itemLevel) ~
      ("maxCount" -> item.maxCount) ~
      ("percent" -> item.discount) ~
      (OfferOps.levelPlural(item.itemLevel) ->
        ("name" -> item.name) ~
        ("ids" -> item.ids))
    )

  val condition: JObject = Nil
  override val recs: JObject = ("rewards" -> this.reward) ~ ("conditions" -> this.condition)
}
//
//abstract case class ProgressiveOffer(scale: Array[Double], levels: Array[Int], validDays: Int) extends Offer {
//  val fmt: JObject =
//    ("type")
//}


object OfferOps {
  def levelPlural(level: String): String = level match {
    case "product" => "products"
    case "category" => "categories"
  }
}