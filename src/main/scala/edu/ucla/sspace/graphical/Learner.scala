package edu.ucla.sspace.learner

import scalala.tensor.sparse.SparseVectorRow

trait Learner {
   def train(data: List[SparseVectorRow[Double]],
             numGroups: Int) : Array[Int]
}
