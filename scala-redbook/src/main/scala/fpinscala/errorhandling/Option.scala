package fpinscala.errorhandling

import java.util.regex._

sealed trait Option[+A] {
  def map[B](f: A => B): Option[B] = this match {
    case None => None
    case Some(x) => Some(f(x))
  }
  def map2_1[B, C](other: Option[B])(f: (A, B) => C): Option[C] = this match {
    // with pattern matching
    case None => None
    case Some(x) => other match {
      case None => None
      case Some(y) => Some(f(x, y))
    }
  }
  def map2_2[B, C](other: Option[B])(f: (A, B) => C): Option[C] =
    // with for comprehension
    for {
      x <- this
      y <- other
    } yield f(x, y)

  def map2_3[B, C](other: Option[B])(f: (A, B) => C): Option[C] =
    // with map and flatMap
    this flatMap (x => other map (y => f(x, y)))

  def flatMap[B](f: A => Option[B]): Option[B] =
    map(f).getOrElse(None)
  def getOrElse[B >: A](default: => B): B = this match {
    case None => default
    case Some(x) => x
  }
  def orElse[B >: A](ob: => Option[B]): Option[B] =
    map(Some(_)).getOrElse(ob)
  def filter(f: A => Boolean): Option[A] =
    flatMap(x => if (f(x)) Some(x) else None)
  }
case class Some[+A](get: A) extends Option[A]
case object None extends Option[Nothing]

object Option {
  def mean(xs: Seq[Double]): Option[Double] =
    if (xs.isEmpty) None
    else Some(xs.sum / xs.length)

  def variance(xs: Seq[Double]): Option[Double] =
    mean(xs) flatMap (m => mean(xs map (x => math.pow(x - m, 2))))

  def pattern(s: String): Option[Pattern] =
    try {
      Some(Pattern.compile(s))
    } catch {
      case e: PatternSyntaxException => None
    }

  def bothMatch(p1: String, p2: String, s: String): Option[Boolean] =
    pattern(p1).map2_3(pattern(p2))((x, y) => x.matcher(s).matches && y.matcher(s).matches)

  def sequence[A](a: List[Option[A]]): Option[List[A]] = a match {
    case Nil => Some(Nil)
    case h :: t => h flatMap (x => sequence(t) map (x :: _))
  }

  def traverse[A, B](a: List[A])(f: A => Option[B]): Option[List[B]] = a match {
    case Nil => Some(Nil)
    case h :: t => f(h).map2_3(traverse(t)(f))(_::_)
  }

  def sequenceViaTraverse[A](a: List[Option[A]]): Option[List[A]] =
    traverse(a)(x => x)
}
