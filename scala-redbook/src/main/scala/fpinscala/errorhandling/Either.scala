package fpinscala.errorhandling

sealed trait Either[+E, +A]{
  def map[B](f: A => B): Either[E, B] = this match {
    case Left(e) => Left(e)
    case Right(a) => Right(f(a))
  }
  def flatMap[EE >: E, B](f: A => Either[EE, B]): Either[EE, B] = this match {
    case Left(e) => Left(e)
    case Right(a) => f(a)
  }
  def orElse[EE >: E,B >: A](b: => Either[EE, B]): Either[EE, B] = this match {
    case Left(_) => b
    case Right(a) => Right(a)
  }
  def map2[EE >: E, B, C](b: Either[EE, B])(f: (A, B) => C): Either[EE, C] =
    flatMap(x => b map (y => f(x, y)))
}

case class Left[+E](value: E) extends Either[E, Nothing]
case class Right[+A](value: A) extends Either[Nothing, A]

object Either {
  def traverse[EE, A, B](a: List[A])(f: A => Either[EE, B]): Either[EE, List[B]] = a match {
    case Nil => Right(Nil)
    case h :: t => f(h).map2(traverse(t)(f))(_::_)
  }

  def sequenceViaTraverse[EE, A](a: List[Either[EE, A]]): Either[EE, List[A]] =
    traverse(a)(x => x)
}
