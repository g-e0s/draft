package setmachine.recsys

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.{Word2Vec,Word2VecModel}

object Model {
  val model: Word2VecModel = ???
  def fit(data: DataFrame): Word2VecModel = {
    val model = new Word2Vec()
    model.fit(data)
  }


}
