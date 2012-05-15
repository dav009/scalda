package edu.ucla.sspace.graphical

import scalala.tensor.mutable.VectorRow

trait Learner {
   def train(data: List[VectorRow[Double]], numGroups: Int) : Array[Int]
}
