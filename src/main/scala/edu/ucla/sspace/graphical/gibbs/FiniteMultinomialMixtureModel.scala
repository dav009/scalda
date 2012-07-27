package edu.ucla.sspace.graphical.gibbs

import edu.ucla.sspace.graphical.Learner

import breeze.numerics._
import breeze.linalg.DenseVector
import breeze.linalg.Vector

import breeze.stats.distributions.Dirichlet
import breeze.stats.distributions.Multinomial

import scala.math.pow


class FiniteMultinomialMixtureModel(val numIterations: Int,
                                    alpha: DenseVector[Double],
                                    gamma: DenseVector[Double],
                                    s: Set[Int] = Set[Int]()) extends Learner {

    val pi_dist = Dirichlet(alpha)

    def train(data: List[Vector[Double]], 
              k: Int,
              ignored: List[List[Vector[Double]]]) = {
        val v = data(0).length
        val n = data.size

        var labelStats = Array.fill(k)(new DenseVector[Double](
            Array.fill(v)(0d)))
        var thetas = sampleThetas(labelStats)

        var pi = pi_dist.sample
        val labels = new Multinomial(pi).sample(n).toArray
        var labelCounts = new DenseVector[Double](Array.fill(k)(0d))
        data.zip(labels).foreach{
            case (x_j, j) => {labelStats(j) += x_j; labelCounts(j) += 1} }

        if (s contains 0)
            report(0, labels.toList)

        for (i <- 0 until numIterations) {
            printf("Iteration [%d]\n", i)
            for ( (x_j, j) <- data.zipWithIndex ) {
                val label = labels(j)
                labelStats(label) -= x_j
                labelCounts(label) -= 1

                val prior = labelCounts + alpha - 1 / (n-1+alpha.sum)
                val posterior = new DenseVector[Double](thetas.map( theta =>
                        x_j.activeIterator
                           .map{ case(i, n_xj) => pow(theta(i), n_xj) }
                           .product).toArray)

                val probs = prior :* posterior
                val new_label= new Multinomial(probs / probs.sum).sample

                labelCounts(new_label) += 1
                labelStats(new_label) += x_j
                labels(j) = new_label
            }
            thetas = sampleThetas(labelStats)
            thetas.foreach(println)

            if (s contains (i+1))
                report(i+1, labels.toList)
        }

        labels
    }

    def sampleThetas(labelStats: Array[DenseVector[Double]]) =
        labelStats.map(n_c => new Dirichlet(n_c+gamma).sample)
}
