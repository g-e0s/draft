package setmachine.model

import org.apache.spark.ml._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions.{avg, broadcast, collect_list}

sealed trait Vectorizer[M] {
  def getItemVectors(data: Dataset[_]): Dataset[_]
}

abstract class ItemVectorizer[M](estimator: Estimator[M],
                                 val itemCol: String,
                                 val userCol: String) extends Vectorizer[M] {
  def fit(data: Dataset[_]): M = estimator.fit(data)
  def getItemVectors(data: Dataset[_]): Dataset[_]
  def preprocess(data: Dataset[_], itemCol: String, userCol: String, basketCol: String): Dataset[_]
}

abstract class UserVectorsDecorator[M](vectorizer: ItemVectorizer[M]) extends Vectorizer[M] {
  override def getItemVectors(data: Dataset[_]): Dataset[_] = vectorizer.getItemVectors(data)
  def getItemUserVectors(data: Dataset[_], itemCol: String, userCol: String, basketCol: String): (Dataset[_], Dataset[_])
}

class Word2VecVectorizer(estimator: Estimator[feature.Word2VecModel],
                         itemCol: String,
                         userCol: String)
  extends ItemVectorizer(estimator: Estimator[feature.Word2VecModel], itemCol: String, userCol: String) {
  def getItemVectors(model: feature.Word2VecModel): Dataset[_] =
    model.getVectors.withColumnRenamed("word", itemCol)
  def getItemVectors(data: Dataset[_]): Dataset[_] =
    fit(data).getVectors.withColumnRenamed("word", itemCol)
  def preprocess(data: Dataset[_], itemCol: String, userCol: String, basketCol: String): Dataset[_] =
    data.groupBy(userCol).agg(collect_list(itemCol).alias(basketCol))
}

class SimpleAverageUserVectorsDecorator[M](vectorizer: ItemVectorizer[M])
  extends UserVectorsDecorator[M](vectorizer: ItemVectorizer[M]) {
  def getItemUserVectors(data: Dataset[_], itemCol: String, userCol: String, basketCol: String): (Dataset[_],Dataset[_]) = {
    val preprocessed = vectorizer.preprocess(data, itemCol, userCol, basketCol)
    val itemVectors = vectorizer.getItemVectors(data)
    val userVectors = data.join(broadcast(data), usingColumn=vectorizer.itemCol)
      .groupBy(vectorizer.userCol)
      .agg(avg("vector").alias("vector"),collect_list(vectorizer.itemCol).alias("basket"))
    (itemVectors, userVectors)
  }
}

object Vectorizer {
  val spark: SparkSession = SparkSession.builder().getOrCreate()
  def explode(baskets: Dataset[_], items: Array[String]): DataFrame = {
    def unpack: Row => TraversableOnce[Row] = {
      case Row(k: String, v: Array[String]) => items.filter(x => !v.contains(x)).map(Row(k, _))
    }
    val schema = StructType(Array(
      StructField(userCol, StringType()),
      StructField(itemCol, StringType())))
    spark.createDataFrame(baskets.rdd.flatMap(unpack), schema=schema)
  }
}