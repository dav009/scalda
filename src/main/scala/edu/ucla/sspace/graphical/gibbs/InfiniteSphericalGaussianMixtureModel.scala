package edu.ucla.sspace.graphical.gibbs

import edu.ucla.sspace.graphical.ComponentGenerator
import edu.ucla.sspace.graphical.Learner
import edu.ucla.sspace.graphical.Likelihood._
import edu.ucla.sspace.graphical.Util.norm

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow

import scalanlp.stats.distributions.Multinomial


class InfiniteSphericalGaussianMixtureModel(val numIterations: Int, 
                                            val alpha: Double,
                                            val generator: ComponentGenerator,
                                            s: Set[Int] = Set[Int]()) extends Learner {

    /**
     * Number of points in component,
     * mean
     * variance
     * assigned points
     */
    type Theta = (Double, DenseVectorRow[Double], Double)

    def train(data: List[VectorRow[Double]], ignored: Int) = {
        val n = data.size
        val t = n - 1 + alpha

        // Compute the global mean of all data points.
        val mu_0 = data.reduce(_+_).toDense / n
        // Compute the variance of all data points to the global mean.
        val variance_0 = data.map(_-mu_0).map(_.norm(2)).map(pow(_, 2)).reduce(_+_) / n

        // Create the global component that will be used to determine when a 
        // new component should be sampled.
        //val components = Array((alpha, mu_0, variance_0)).toBuffer
        val components = Array((alpha, mu_0, variance_0)).toBuffer

        // Setup the initial labels for all the data points.  These start off 
        // with no meaningful value.
        var labels = Array.fill(n)(0)

        for (i <- 0 until numIterations) {
            printf("Starting iteration [%d] with [%d] components,\n", i, components.size-1)
            for ( (x_j, j) <- data.zipWithIndex ) {
                // Setup a helper function to compute the likelihood for point x.
                def likelihood(theta: Theta) = gaussian(x_j, theta._2, theta._3)

                val l_j = labels(j)

                // Undo the assignment for the existing point.  This involes
                // first removing the information from the variance vectors,
                // then removing it from the center vector, then finally undoing
                // the count for the component.  Save the original component 
                // data so we can restore it quickly later on.
                if (i != 0)
                    components(l_j) = updateComponent(components(l_j), x_j, -1)

                // Compute the probability of selecting each component based on
                // their sizes.
                val prior = DenseVectorRow[Double](components.map(_._1 / t).toArray)
                // Compute the probability of the data point given each
                // component using the sufficient statistics.
                val posterior = DenseVectorRow[Double](components.map(likelihood).toArray)
                val probs = norm(prior :* posterior)

                // Combine the two probabilities into a single distribution and
                // select a new label for the data point.
                val l_j_new  = new Multinomial(probs).sample

                if (l_j_new == 0) {
                    // If the global component was created, create a new 
                    // component using just the current data point.
                    labels(j) = components.size
                    components.append(generator.sample(List(x_j), 1d))
                } else {
                    // Restore the bookeeping information for this point using the
                    // old assignment.
                    labels(j) = l_j_new
                    components(l_j_new) = updateComponent(components(l_j_new), x_j, 1)
                }
            }

            val sigmas = components.map(_._3)
            // Re-estimate the means, counts, and variances for each component.
            // We do this by first grouping the data points based on their
            // assigned component, summing the points assigned to each
            // component, and finally computing the variance of each point from
            // the mean.
            val labelRemap = labels.zip(data)
                                   .groupBy(_._1)
                                   .map{ case(k,v) => (k, v.map(_._2)) }
                                   .zipWithIndex
                                   .map{ case ((c_old, x), c) => {
                components(c+1) = generator.sample(x.toList, sigmas(c_old))
                (c_old, c+1)
            }}
            components.trimEnd(components.size - (labelRemap.size+1))
            labels = labels.map(labelRemap)
            val mu_k = components.view(1, components.size).map(_._2).toArray
            val variance_k = components.view(1, components.size).map(_._3).toArray
            generator.update(mu_k, variance_k)

            if (s contains (i+1))
                report(i+1, labels.toList)
        }

        // Return the labels.
        labels.toArray
    }

    def updateComponent(theta: Theta, x: VectorRow[Double], delta: Double) =
        if (delta >= 0)
            (theta._1+delta, theta._2, theta._3)
        else
            (theta._1+delta, theta._2, theta._3)
}
