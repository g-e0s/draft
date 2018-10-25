package setmachine.profiles

import breeze.linalg.DenseVector
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.scalatest.FunSuite
import org.apache.spark.sql.{DataFrame, Row, SparkSession, functions}

object VectorizerTest extends FunSuite {
  val vectorizer: Vectorizer = new Vectorizer()
  val sc: SparkContext = vectorizer.spark.sparkContext
  test("CubeCalculator.cube") {
    val schema = StructType(Array(
      StructField("clientGUID", StringType()),
      StructField("item", StringType()))
    )
    val data: RDD[Row] = sc.makeRDD(Array(
      Row("user1", "item1", DenseVector(0.0, 0.3, 0.5)),
      Row("user1", "item1"),
      Row("user1", "item1")))
    val df: DataFrame = vectorizer.spark.createDataFrame(data, schema)
    //assert(Vectorizer.)
  }
}
