package scalars.lib.offer

sealed trait Discount
case class ItemDiscount(name: String, itemLevel: String, ids: List[String], discount: Double, maxCount: Int,
                              conditions: List[Condition])
