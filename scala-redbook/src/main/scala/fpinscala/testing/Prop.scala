package fpinscala.testing

import fpinscala.testing.Prop.{FailedCase, SuccessCount}

trait Prop {
  def check: Either[FailedCase, SuccessCount]
  def && (p: Prop): Boolean = check.isRight && p.check.isRight
  def forAll[A](a: Gen[A])(f: A => Boolean): Prop
}

object Prop {
  type FailedCase = String
  type SuccessCount = Int
}
