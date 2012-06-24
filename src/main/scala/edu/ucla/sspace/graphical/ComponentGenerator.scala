package edu.ucla.sspace.graphical

import breeze.linalg.DenseVector
import breeze.linalg.Vector


trait ComponentGenerator {
    def initial() : (Double, DenseVector[Double], Double)
    def initialMean() : DenseVector[Double]
    def initialVariance() : Double
    def sample(data: List[Vector[Double]], sigma_2_old: Double) : (Double, DenseVector[Double], Double)
    def update(mu_k: Array[DenseVector[Double]], variance_k: Array[Double])
}
