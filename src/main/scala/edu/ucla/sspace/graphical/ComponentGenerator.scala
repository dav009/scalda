package edu.ucla.sspace.graphical

import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow


trait ComponentGenerator {
    def sample(data: List[VectorRow[Double]], sigma_2_old: Double) : (Double, DenseVectorRow[Double], Double)
}
