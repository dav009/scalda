package edu.ucla.sspace.graphical.gibbs

import edu.ucla.sspace.graphical.Learner
import edu.ucla.sspace.graphical.Likelihood.gaussian
import edu.ucla.sspace.graphical.Util.{norm,epsilon}

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow

import scalanlp.stats.distributions.Multinomial

import scala.math.Pi


class FiniteGaussianMixtureModel(val numIterations: Int, 
                                 val alpha: Double,
                                 val useKMeans: Boolean = false) extends Learner {

    def train(data: List[VectorRow[Double]], k: Int) = {
        // Extract the shape of the data.
        val n = data.size
        val v = data(0).length

        // Assign random labels to all points.
        val labels = new Multinomial(Array.fill(k)(1d/k)).sample(n).toArray

        // Compute the global mean of all data points.
        val mu_1 = data.reduce(_+_).toDense / n
        // Compute the variance of all data points to the global mean.
        val variance_1 = (data.map(x_j => (mu_1 - x_j) :^ 2).reduce(_+_) / n) + epsilon
        // Compute the standard deviation to the global mean.
        val sigma_1 = variance_1 :^ 0.5

        // Sample from a standard normal distribution to create initial means
        // for each component.  These components also have a standard variance.
        var mus = Array.fill(k)(DenseVectorRow.randn(v) :* sigma_1 :+ mu_1)
        var sigmas = Array.fill(k)(DenseVectorRow.ones[Double](v))

        // Compute the number of assignments to each component and the summation
        // vector for each component.
        var counts = DenseVectorRow.ones[Double](k)

        for (i <- 0 until numIterations) {
            printf("Iteration [%d]\n", i)
            for ( (x_j, j) <- data.zipWithIndex ) {
                // Undo the assignment for the existing point.  This involes
                // first removing the information from the variance vectors,
                // then removing it from the center vector, then finally undoing
                // the count for the component.
                val l_j = labels(j)
                if (i != 0) {
                    sigmas(l_j) -= (mus(l_j)/counts(l_j) - x_j) :^ 2
                    mus(l_j) -= x_j
                    counts(l_j) -= 1
                }

                // Compute the probability of selecting each component based on
                // their sizes.
                val prior = (counts + alpha/k ) / (n - 1 + alpha)
                // Compute the probability of the data point given each
                // component using the sufficient statistics.
                val posterior = new DenseVectorRow[Double](
                    (counts.data, mus, sigmas).zipped.toList.map {
                        case (n_c, mu_c, sigma_c) => 
                            if (n_c < 1d) 0d
                            else gaussian(x_j, mu_c/n_c, sigma_c/n_c) }.toArray) + epsilon

                val probs = norm(prior :* posterior)

                // Combine the two probabilities into a single distribution and
                // select a new label for the data point.  Record this in the
                // label array.
                labels(j) = if (useKMeans) probs.argmax
                            else new Multinomial(probs).sample

                if (i != 0) {
                    // Restore the bookeeping information for this point using the
                    // old assignment.
                    counts(l_j) += 1
                    mus(l_j) += x_j
                    sigmas(l_j) += (mus(l_j)/counts(l_j) - x_j) :^ 2
                }
            }

            // Re-estimate the means, counts, and variances for each component.
            // We do this by first grouping the data points based on their
            // assigned component, summing the points assigned to each
            // component, and finally computing the variance of each point from
            // the mean.
            labels.zip(data).groupBy(_._1)
                            .map{ case(k,v) => (k, v.map(_._2)) }
                            .foreach{ case(c, x) => {
                mus(c) = x.reduce(_+_).toDense
                counts(c) = x.size
                sigmas(c) = x.map(x_j => (mus(c)/counts(c) - x_j) :^2).reduce(_+_) + epsilon
            }}
        }

        // Return the labels.
        labels
    }
}
