package edu.ucla.sspace.graphical.gibbs

import edu.ucla.sspace.graphical.Learner
import edu.ucla.sspace.graphical.Util.{norm,sampleUnormalizedLogMultinomial}

import scalala.library.Library._
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.dense.DenseVectorRow
import scalala.tensor.mutable.VectorRow

import scalanlp.stats.distributions.Beta
import scalanlp.stats.distributions.Dirichlet
import scalanlp.stats.distributions.Multinomial

import scala.io.Source


class InfiniteMultinomialMixtureModel(val numIterations: Int,
                                      alpha: Double,
                                      gamma: Double,
                                      s: Set[Int] = Set[Int]()) extends Learner {

    type Theta = (Double, DenseVectorRow[Double], DenseVectorRow[Double])

    def train(data: List[VectorRow[Double]],
              ignored: Int,
              priorData: List[List[VectorRow[Double]]]) = {
        val v = data(0).length
        val n = data.size

        val globalStats = data.reduce(_+_).toDense
        var components = Array((alpha, sampleTheta(globalStats), globalStats)).toBuffer
        priorData.foreach( preGroup => {
                val cummulative = preGroup.reduce(_+_).toDense
                components.append((preGroup.size.toDouble, sampleTheta(cummulative), cummulative))
        })

        var labels = Array.fill(n)(0)
        val lines = Source.stdin.getLines

        for (i <- 0 until numIterations) {
            printf("Starting iteration [%d] with [%d] components,\n", i, components.size-1)
            printf("Sampling new assignmentions\n")
            for ( (x_j, j) <- data.zipWithIndex ) {
                def prior(n_c: Double) = 
                    n_c / (if (i == 0) (j+alpha) else (n-1+alpha))
                def likelihood(theta: DenseVectorRow[Double]) =
                    x_j.pairsIteratorNonZero
                       .map{ case (k, v) => v * log (theta(k)) }
                       .sum
                    /*
                       .map{ case (k, v) => pow(theta(k), v) }
                       .product
                       */

                val l_j = labels(j)
                if (i != 0)
                    components(l_j) = update(components(l_j), x_j, -1)

                val probs = components.map( c => log(prior(c._1)) + likelihood(c._2))
                                      .toArray
                /*
                val probs = DenseVectorRow[Double](
                    components.map( c => prior(c._1) * likelihood(c._2))
                              .toArray)

                println(components.map(c => 
                    x_j.pairsIteratorNonZero
                       .map{ case (k, v) => v * log(c._2(k)) }
                       .sum).mkString(" "))
                */

                val l_j_new = sampleUnormalizedLogMultinomial(probs)
                println(probs.mkString(" "))
                println(l_j_new)
                print("Step forward:")
                lines.next
                    //new Multinomial(norm(probs)).sample

                if (l_j_new == 0) {
                    labels(j) = components.size
                    components.append(newComponent(x_j))
                } else {
                    labels(j) = l_j_new
                    components(l_j_new) = update(components(l_j_new), x_j, 1)
                }
            }

            printf("Sampling new component parameters\n")
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

    def update(theta: Theta, x: VectorRow[Double], diff: Int) = {
        val newCumSum = theta._3
        x.foreachNonZeroPair( (i,v) => {
            if (diff >= 0)
                newCumSum(i) += v
            else
                newCumSum(i) -= v
        })

        if (diff >= 0)
            (theta._1+diff, theta._2, newCumSum)
        else
            (theta._1+diff, theta._2, newCumSum)
    }

    def newComponent(x: VectorRow[Double]) =
        (1d, sampleTheta(x), x.toDense)

    def sampleTheta(x: VectorRow[Double]) =
        new Dirichlet(x.toDense + gamma).sample
}
