package setmachine.profiles

import breeze.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.{Word2Vec, Normalizer}

class Vectorizer {
  val userCol: String = "clientGUID"
  val itemCol: String = "groupCode"
  val inputCol: String = "basket"
  val outputCol: String = "vector"
  val similarityCol: String = "similarity"
  val leftVectorName: String = "leftVector"
  val rightVectorName: String = "rightVector"
  val spark: SparkSession = SparkSession.builder().getOrCreate()
  val normalizer: Normalizer = new Normalizer()
    .setInputCol(outputCol)
    .setOutputCol(outputCol)
  def vectorize(baskets: DataFrame, vectorSize: Int, windowSize: Int): (DataFrame, DataFrame) = {
    val model = new Word2Vec()
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setVectorSize(vectorSize)
      .setWindowSize(windowSize)
      .fit(baskets)
    val itemVectors = normalizer.transform(model.getVectors)
    val userVectors = normalizer.transform(model.transform(baskets))
    (itemVectors, userVectors)
  }

  def getBaskets(data: DataFrame): DataFrame =
    data.groupBy(userCol).agg(collect_list(itemCol).alias(inputCol))

  def getAbsentPairs(baskets: DataFrame, items: Array[String]): DataFrame = {
    def unpack: Row => TraversableOnce[Row] = {
      case Row(k: String, v: Array[String]) => items.filter(x => !v.contains(x)).map(Row(k, _))
    }
    val schema = StructType(Array(
      StructField(userCol, StringType()),
      StructField(itemCol, StringType())))
    spark.createDataFrame(baskets.rdd.flatMap(unpack), schema=schema)
  }

  def cosineSimilarity(u: DenseVector[Double], v: DenseVector[Double]): Double = u.dot(v)

  def calculateSimilarity(pairs: DataFrame): DataFrame = {
    def mapper: Row => Row = {
      case Row(wordLeft: String, wordRight: String,
      vectorLeft: DenseVector[Double], vectorRight: DenseVector[Double])=>
        Row(wordLeft, wordRight, cosineSimilarity(vectorLeft, vectorRight))
    }
    val schema = StructType(Array(
      StructField("wordLeft", StringType()),
      StructField("wordRight", StringType()),
      StructField("similarity", DoubleType()))
    )
    spark.createDataFrame(pairs.rdd.map(mapper), schema)
  }
}
