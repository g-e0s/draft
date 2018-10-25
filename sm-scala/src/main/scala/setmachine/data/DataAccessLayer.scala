package setmachine.data

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

trait IData[A] {
  def map[B](f: A => B)
  def flatMap[B](f: A => TraversableOnce[B])
  def filter[Boolean](f: A => Boolean)
  def distinct: A
  def addField[B](name: String, f: A => B)
  def dropField(name: String)

  def join[B](other: B, on: Array[String], how: String)
  def union(other: A)
  def except(other: A)
}

case class DataRDD[A, B](data: RDD[A]) extends IData[A] {
  def map(f: A => B): RDD[B] = data.map(f)
  def flatMap(f: A => TraversableOnce[B]): RDD[B] = data.flatMap(f)
  def filter(f: A => Boolean): RDD[A] = data.filter(f)
  def distinct: RDD[A] = data.distinct()
  def addField(name: String, f: A => B)
  def dropField(name: String)
}