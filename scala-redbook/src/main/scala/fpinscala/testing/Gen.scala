package fpinscala.testing

import fpinscala.state._
import fpinscala.laziness.Stream

case class Gen[+A](sample: State[RNG,A], exhaustive: Stream[Option[A]])

object Gen {
  //type Gen[A] = State[RNG, A]
  def unit[A](a: => A): Gen[A] =
    Gen(State.unit(a), Stream.empty)
  def boolean: Gen[Boolean] =
    Gen(State(RNG.boolean), Stream(Some(true), Some(false)))
  def choose(start: Int, stopExclusive: Int): Gen[Int] =
    Gen(State(RNG.choose(start, stopExclusive)), Stream.from(start).takeWhile(_ < stopExclusive).map(Some(_)))
  /** Generate lists of length n, using the given generator. */
  def listOfN[A](n: Int, g: Gen[A]): Gen[List[A]] = ???
    // g.exhaustive.take(n).map(_.getOrElse(g.sample))
  /** Between 0 and 1, not including 1. */
  def uniform: Gen[Double] =
    Gen(State(RNG.double), Stream.empty)
  /** Between `i` and `j`, not including `j`. */
  def choose(i: Double, j: Double): Gen[Double] =
    Gen(State(RNG.map(RNG.double)(x => i + x * (j-i))), Stream.empty)
}
