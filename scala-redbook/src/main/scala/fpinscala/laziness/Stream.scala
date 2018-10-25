package fpinscala.laziness

sealed trait Stream[+A] {
  def uncons: Option[(A, Stream[A])]

  def isEmpty: Boolean = uncons.isEmpty

  def toList: List[A] = {
    @annotation.tailrec
    def go(s: Stream[A], acc: List[A]): List[A] =
      s.uncons match {
        case None => acc
        case Some((h, t)) => go(t, acc :+ h)
      }
    go(this, List())
  }

  def take(n: Int): Stream[A] =
    uncons match {
      case Some((h, t)) => Stream.cons(h, if (n > 1) t.take(n - 1) else Stream.empty)
      case None => this
    }

  def takeWhile(p: A => Boolean): Stream[A] =
    uncons match {
      case Some((h, t)) => if (p(h)) Stream.cons(h, t.takeWhile(p)) else Stream.empty
      case None => this
    }

  def foldRight[B](z: => B)(f: (A, => B) => B): B =
    uncons match {
      case Some((h, t)) => f(h, t.foldRight(z)(f))
      case None => z
    }

  def exists(p: A => Boolean): Boolean = foldRight(false)((x, y) => p(x) || y)
  def forall(p: A => Boolean): Boolean = foldRight(true)((x, y) => p(x) && y)

  def takeWhileViaFoldRight(p: A => Boolean): Stream[A] =
    foldRight(Stream.empty: Stream[A])((x, y) => if (p(x)) Stream.cons(x, y) else y)

  def map[B](f: A => B): Stream[B] =
    foldRight(Stream.empty: Stream[B])((x, y) => Stream.cons(f(x), y))

  def append[B >: A](s: => Stream[B]): Stream[B] =
    foldRight(s)((x, y) => Stream.cons(x, y))

  def flatMap[B](f: A => Stream[B]): Stream[B] =
    foldRight(Stream.empty: Stream[B])((x, y) => f(x) append y)

  def filter(f: A => Boolean): Stream[A] =
    foldRight(Stream.empty: Stream[A])((x, y) => if (f(x)) Stream.cons(x, y) else y)

  def mapViaUnfold[B](f: A => B): Stream[B] =
    Stream.unfold(this.uncons)(
      {
        case Some((h, t)) => Some((f(h), t.uncons))
        case None => None
      }
    )

  def takeViaUnfold(n: Int): Stream[A] =
    Stream.unfold((this.uncons, n))(
      {
        case (Some((h, _)), 1) => Some((h, (None, 0)))
        case (Some((h, t)), m) => Some((h, (t.uncons, m - 1)))
        case (None, _) => None
      }
    )

  def takeWhileViaUnfold(p: A => Boolean): Stream[A] =
    Stream.unfold(this.uncons)(
      {
        case Some((h, t)) => if (p(h)) Some((h, t.uncons)) else None
        case None => None
      }
    )

  def zip[B](other: Stream[B]): Stream[(A,B)] =
    zipWith(other)((_, _))

  def zipWith[B,C](other: Stream[B])(f: (A,B) => C): Stream[C] =
    Stream.unfold((this.uncons, other.uncons))(
      {
        case (Some((h0, t0)), Some((h1, t1))) => Some((f(h0, h1), (t0.uncons, t1.uncons)))
        case (_, _) => None
      }
    )
  def zipAll[B](other: Stream[B]): Stream[(Option[A],Option[B])] =
    Stream.unfold((this.uncons, other.uncons))(
      {
        case (Some((h0, t0)), Some((h1, t1))) => Some(((Some(h0), Some(h1)), (t0.uncons, t1.uncons)))
        case (Some((h0, t0)), None) => Some(((Some(h0), None), (t0.uncons, None)))
        case (None,  Some((h1, t1))) => Some(((None, Some(h1)), (None, t1.uncons)))
        case (None, None) => None
      }
    )

  def tails: Stream[Stream[A]] =
    Stream.unfold(this.uncons)(
      {
        case Some((h, t)) => Some((Stream.cons(h, t), t.uncons))
        case None => None
      }
    )

  def scanRight[B](z: => B)(f: (A, B) => B): Stream[B] =
    Stream.unfold((this.uncons, z))(
      {
        case (Some((h, t)), zz) => Some((f(h, zz), (t.uncons, f(h, zz))))
        case (None, _) => None
      }
    )
}

object Stream {
  def empty[A]: Stream[A] =
    new Stream[A] {def uncons: Option[(A, Stream[A])] = None}
  def cons[A](h: => A, t: => Stream[A]): Stream[A] =
    new Stream[A] {lazy val uncons: Option[(A, Stream[A])] = Some((h, t))}
  def unfold[A, S](z: S)(f: S => Option[(A, S)]): Stream[A] = f(z) match {
    case Some((a, zz)) => cons(a, unfold(zz)(f))
    case None => Stream.empty
  }
  def apply[A](as: A*): Stream[A] =
    if (as.isEmpty) empty
    else cons(as.head, apply(as.tail : _*))
  def constant[A](c: => A): Stream[A] = cons(c, constant(c))
  def from(n: Int): Stream[Int] = cons(n, from(n+1))
  def fibs: Stream[Int] = {
    def go(a: Int, b: Int): Stream[Int] = cons(a, go(b, a + b))
    go(0, 1)
  }
  def constantViaUnfold[A](c: => A): Stream[A] =
    unfold(c)(_ => Some((c,c)))
  def fromViaUnfold(n: Int): Stream[Int] =
    unfold(n)(n => Some((n, n + 1)))
  def fibsViaUnfold: Stream[Int] =
    unfold((0, 1))({case (f0, f1) => Some((f0, (f1, f0+f1)))})

  def startsWith[A](s1: Stream[A], s2: Stream[A]): Boolean =
    s1.zipAll(s2).takeWhile(!_._2.isEmpty).foldRight(true)((x, y) => (x._1 == x._2) && y)

  def hasSubsequence[A](s1: Stream[A], s2: Stream[A]): Boolean =
    s1.tails.exists(startsWith(_, s2))
}