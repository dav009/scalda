package edu.ucla.sspace.graphical.gibbs

import edu.ucla.sspace.graphical.Learner

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow

import scalanlp.stats.distributions.Beta
import scalanlp.stats.distributions.Dirichlet
import scalanlp.stats.distributions.Multinomial


class GibbsNaiveBayes(val numIterations: Int,
                      gamma_pi: DenseVectorRow[Double],
                      gamma_theta: DenseVectorRow[Double]) extends Learner {

    val pi_dist = new Dirichlet(gamma_pi)

    def train(data: List[VectorRow[Double]], 
              k: Int) = {
        val v = data(0).length
        val n = data.size

        var labelStats = Array.fill(k)(new DenseVectorRow[Double](
            Array.fill(v)(0d)))
        var thetas = sampleThetas(labelStats)

        var pi = pi_dist.sample
        val labels = new Multinomial(pi.data).sample(n).toArray
        var labelCounts = new DenseVectorRow[Double](Array.fill(k)(0d))
        data.zip(labels).foreach{
            case (x_j, j) => {labelStats(j) += x_j; labelCounts(j) += 1} }

        for (i <- 0 until numIterations) {
            printf("Iteration [%d]\n", i)
            for ( (x_j, j) <- data.zipWithIndex ) {
                val label = labels(j)
                labelStats(label) -= x_j
                labelCounts(label) -= 1

                val prior = labelCounts + gamma_pi - 1 / (n-1+gamma_pi.sum)
                val posterior = new DenseVectorRow[Double](thetas.map( theta =>
                    (theta :^ x_j).iterator.product).toArray)

                val probs = prior :* posterior
                val new_label= new Multinomial(probs / probs.sum).sample

                labelStats(new_label) += x_j
                labelCounts(new_label) += 1
                labels(j) = new_label
            }
            thetas = sampleThetas(labelStats)
        }

        labels
    }

    def sampleThetas(labelStats: Array[DenseVectorRow[Double]]) =
        labelStats.map(n_c => new Dirichlet(n_c+gamma_theta).sample)
}
