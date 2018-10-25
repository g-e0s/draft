package setmachine.recommender

import org.apache.spark.sql.Dataset

trait Recommender

abstract class SimilarityBasedRecommender(itemVectors: Dataset[_], userVectors: Dataset[_]) extends Recommender {
  def
}
