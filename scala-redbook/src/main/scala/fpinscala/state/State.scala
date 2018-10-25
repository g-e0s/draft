package fpinscala.state

case class State[S,+A](run: S => (A,S)) {
  def flatMap[B](f: A => State[S, B]): State[S, B] =
    State(s => {
      val (a, s0) = run(s)
      f(a).run(s0)
    })

  def map[B](f: A => B): State[S, B] =
    flatMap(s => State.unit(f(s)))

  def map2[B,C](b: State[S, B])(f: (A, B) => C): State[S, C] =
    flatMap(a => b.map(bs => f(a, bs)))

  def get: State[S, S] = State(s => (s, s))

  def set(s: S): State[S, Unit] = State(_ => ((), s))

  def modify(f: S => S): State[S, Unit] = for {
    s <- get
    _ <- set(f(s))
  } yield ()
}

object State {
  def unit[S, A](a: A): State[S, A] =
    State(s => (a, s))

  def sequence[S, A](fs: List[State[S, A]]): State[S, List[A]] =
    fs.foldRight(unit[S, List[A]](List[A]()))((f, acc) => f.map2(acc)(_ :: _))
}
