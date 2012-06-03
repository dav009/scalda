package edu.ucla.sspace.graphical.gibbs

import edu.ucla.sspace.graphical.Learner
import edu.ucla.sspace.graphical.Util.norm

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow

import scalanlp.stats.distributions.Beta
import scalanlp.stats.distributions.Dirichlet
import scalanlp.stats.distributions.Multinomial


class InfiniteMultinomialMixtureModel(val numIterations: Int,
                                      alpha: Double,
                                      gamma: Double,
                                      s: Set[Int] = Set[Int]()) extends Learner {

    type Theta = (Double, DenseVectorRow[Double], DenseVectorRow[Double])

    def train(data: List[VectorRow[Double]], ignored: Int) = {
        val v = data(0).length
        val n = data.size
        def prior(n_c: Double) = n_c / (n-1+alpha)

        val globalStats = data.reduce(_+_).toDense
        var components = Array((alpha, sampleTheta(globalStats), globalStats)).toBuffer
        var labels = Array.fill(n)(0)

        for (i <- 0 until numIterations) {
            printf("Starting iteration [%d] with [%d] components,\n", i, components.size-1)
            for ( (x_j, j) <- data.zipWithIndex ) {
                def likelihood(theta: DenseVectorRow[Double]) =
                    (theta :^ x_j).iterator.product

                val l_j = labels(j)
                if (i != 0)
                    components(l_j) = update(components(l_j), x_j, -1)

                val probs = DenseVectorRow[Double](
                    components.map( c => prior(c._1) * likelihood(c._2))
                              .toArray)
                val l_j_new = new Multinomial(norm(probs)).sample

                if (l_j_new == 0) {
                    labels(j) = components.size
                    components.append(newComponent(x_j))
                } else {
                    labels(j) = l_j_new
                    components(l_j_new) = update(components(l_j_new), x_j, 1)
                }
            }

            val newComponents = Array[Theta]().toBuffer
            val labelRemap = components.zipWithIndex
                                       .filter(_._1._1 > 0)
                                       .zipWithIndex
                                       .map{ case((theta, old_c), new_c) => {
                newComponents.append((theta._1, sampleTheta(theta._3), theta._3))
                (old_c, new_c)
            }}.toMap
            if (components.size != newComponents.size)
                labels = labels.map(labelRemap)
            components = newComponents

            if (s contains (i))
                report(i, labels.toList)
        }

        labels
    }

    def update(theta: Theta, x: VectorRow[Double], diff: Int) =
        if (diff >= 0)
            (theta._1+diff, theta._2, theta._3 + x)
        else
            (theta._1+diff, theta._2, theta._3 - x)

    def newComponent(x: VectorRow[Double]) =
        (1d, sampleTheta(x), x.toDense)

    def sampleTheta(x: VectorRow[Double]) =
        new Dirichlet(x.toDense + gamma).sample
}
