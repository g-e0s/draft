package fpinscala.parallelism

import java.util.concurrent._
import scala.concurrent.duration.TimeUnit

case class UnitFuture[A](a: A) extends Future[A] {
  def get: A = a
  def get(timeout: Long, unit: TimeUnit): A = a
  def cancel(evenIfRunning: Boolean): Boolean = false
  def isDone: Boolean = true
  def isCancelled: Boolean = false
}

object Par {
  type Par[A] = ExecutorService => Future[A]
  def unit[A](a: A): Par[A] = _ => UnitFuture(a)
  def map2[A, B, C](a: Par[A], b: Par[B])(f: (A, B) => C): Par[C] = s => UnitFuture(f(a(s).get, b(s).get))
  // def map2[A, B, C](a: Par[A], b: Par[B])(f: (A, B) => C): Par[C] = s => unit(f(run(s)(a).get, run(s)(b).get))(s)
  def forkBlocking[A](a: => Par[A]): Par[A] = s => s.submit(() => a(s).get) // deadlocks when thread pool size is fixed
  def delay[A](a: => Par[A]): Par[A] = s => a(s) // delay instantiation of a parallel computation until it is actually needed
  def fork[A](a: => Par[A]): Par[A] = forkBlocking(a)
  def async[A](a: => A): Par[A] = fork(unit(a))
  def run[A](s: ExecutorService)(a: Par[A]): Future[A] = a(s)
  def asyncF[A,B](f: A => B): A => Par[B] = a => async(f(a))
  def map[A, B](fa: Par[A])(f: A => B): Par[B] =
    map2(fa, unit(()))((a, _) => f(a))

  def sequence[A](a: List[Par[A]]): Par[List[A]] =
    a.foldRight(unit(List[A]()))((x, y) => map2(x, y)(_ :: _))

  def parMap[A,B](l: List[A])(f: A => B): Par[List[B]] =
    fork(sequence(l.map(asyncF(f))))

  def parFilter[A](l: List[A])(f: A => Boolean): Par[List[A]] =
    map(sequence(l.map(asyncF(a => if (f(a)) List(a) else List()))))(_.flatten)

  def equal[A](a: Par[A], b: Par[A])(s: ExecutorService): Boolean =
    a(s).get == b(s).get

  def choice[A](a: Par[Boolean])(ifTrue: Par[A], ifFalse: Par[A]): Par[A] =
    choiceN(map(a)(x => if (x) 1 else 0))(List(ifFalse, ifTrue))
    // s => if (a(s).get) ifTrue(s) else ifFalse(s)

  def choiceN[A](a: Par[Int])(choices: List[Par[A]]): Par[A] =
    s => choices(a(s).get)(s)

  def choiceMap[A,B](a: Par[A])(choices: Map[A,Par[B]]): Par[B] =
    s => run(s)(choices(a(s).get))

  def flatMap[A, B](a: Par[A])(choices: A => Par[B]): Par[B] =
    s => run(s)(choices(a(s).get))

  def choiceViaFlatMap[A](a: Par[Boolean])(ifTrue: Par[A], ifFalse: Par[A]): Par[A] =
    flatMap(a)(if (_) ifTrue else ifFalse)

  def choiceNViaFlatMap[A](a: Par[Int])(choices: List[Par[A]]): Par[A] =
    flatMap(a)(choices(_))

  def join[A](a: Par[Par[A]]): Par[A] =
    s => run(s)(run(s)(a).get())

  def joinViaFlatMap[A](a: Par[Par[A]]): Par[A] =
    flatMap(a)(x => x)

  def product[A,B](fa: Par[A], fb: Par[B]): Par[(A,B)] = s => unit((run(s)(fa).get, run(s)(fb).get))(s)
  def map_2[A,B](fa: Par[A])(f: A => B): Par[B] = s => unit(f(run(s)(fa).get))(s)
  def map2_2[A,B,C](a: Par[A], b: Par[B])(f: (A, B) => C): Par[C] = map(product(a, b))(x => f(x._1, x._2))
}