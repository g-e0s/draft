package fpinscala.state

sealed trait RNG {
  def nextInt: (Int, RNG)
}


object RNG {
  type Rand[+A] = RNG => (A, RNG)
  // type Rand[+A] = State[RNG, A]
  val int: Rand[Int] = _.nextInt

  def unit[A](a: A): Rand[A] =
    rng => (a, rng)

  def flatMap[A,B](f: Rand[A])(g: A => Rand[B]): Rand[B] =
    rng => {
      val (v, r) = f(rng)
      g(v)(r)
    }

  def map[S, A, B](a: S => (A, S))(f: A => B): S => (B, S) =
    rng => {
      val (v, r) = a(rng)
      (f(v), r)
    }

  def mapViaFlatMap[A, B](a: Rand[A])(f: A => B): Rand[B] =
    flatMap(a)(s => unit(f(s)))

  def positiveMax(n: Int): Rand[Int] =
    map(positiveInt)(x => x * n / Int.MaxValue)

  def choose(start: Int, end: Int): Rand[Int] =
    map(RNG.positiveInt)(x => start + x * (end - start) / Int.MaxValue)

  def simple(seed: Long): RNG = new RNG {
    def nextInt: (Int, RNG) = {
      val seed2 = (seed*0x5DEECE66DL + 0xBL) &
        ((1L << 48) - 1)
      ((seed2 >>> 16).asInstanceOf[Int],
        simple(seed2))
    }
  }
  def positiveInt(rng: RNG): (Int, RNG) = {
    val (x, r) = rng.nextInt
    (if (x < 0) -(x + 1) else x, r)
  }

  def boolean(rng: RNG): (Boolean, RNG) = {
    val (x, r) = rng.nextInt
    (x >= 0, r)
  }

  def double(rng: RNG): (Double, RNG) = {
    val (x, r) = positiveInt(rng)
    (x.toDouble / Int.MaxValue, r)
  }
  def doubleViaMap(rng: RNG): (Double, RNG) =
    map(positiveInt)(_.toDouble / Int.MaxValue)(rng)

  def map2[A,B,C](ra: Rand[A], rb: Rand[B])(f: (A, B) => C): Rand[C] =
    rnd => {
      val (a, r1) = ra(rnd)
      val (b, r2) = rb(r1)
      (f(a, b), r2)
    }


  def map2ViaFlatMap[A, B, C](ra: Rand[A], rb: Rand[B])(f: (A, B) => C): Rand[C] =
    flatMap(ra)(a => map(rb)(b => f(a, b)))

  def intDouble(rng: RNG): ((Int,Double), RNG) = {
    val (i, r1) = positiveInt(rng)
    val (d, r2) = double(r1)
    ((i, d), r2)
  }

  def intDoubleViaMap(rng: RNG): ((Int,Double), RNG) =
    map2(positiveInt, double)((_, _))(rng)

  def doubleInt(rng: RNG): ((Double,Int), RNG) = {
    val ((i, d), r) = intDouble(rng)
    ((d, i), r)
  }
  def sequence[A](fs: List[Rand[A]]): Rand[List[A]] =
    fs.foldRight(unit(List[A]()))((f, acc) => map2(f, acc)(_ :: _))

  def doubles(count: Int)(rng: RNG): (List[Double], RNG) = {
    if (count == 0) (List(), rng)
    else {
      val (d, r) = double(rng)
      val (n, r2) = doubles(count - 1)(r)
      (d :: n, r2)
    }
  }
  def intsViaSequence(count: Int)(rng: RNG): List[Int] =
    sequence(List.fill(count)(int))(rng)._1

  def doublesViaSequence(count: Int)(rng: RNG): (List[Double], RNG) =
    sequence(List.fill(count)(double _))(rng)
}
