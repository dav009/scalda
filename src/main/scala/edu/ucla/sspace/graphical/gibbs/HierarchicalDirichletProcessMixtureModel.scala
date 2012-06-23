package edu.ucla.sspace.graphical.gibbs

import edu.ucla.sspace.graphical.Learner
import edu.ucla.sspace.graphical.Likelihood.gaussian
import edu.ucla.sspace.graphical.Util.{norm,epsilon}

import breeze.numerics._
import breeze.linalg.DenseVector
import breeze.linalg.Vector

import breeze.stats.distributions.Multinomial


class HierarchicalDirichletProcessMixtureModel(val alpha: Double,
                                               val gamma: Double,
                                               val numIter: Int) {

    type Theta = (Double, DenseVectorRow[Double], DenseVectorRow[Double])

    def train(data: List[List[VectorRow[Double]]]) = {
        // Get dimensions for the full dataset.
        val numGroups = data.size
        val numPointsPerGroup = data.map(_.size)
        val numPoints = numPointsPerGroup.sum

        // Create a matrix to hold local table indicator values for each data 
        // point in each group.
        val localLabels = numPointsPerGroup.map(n => Array.fill(n)(0)).toArray
        // Create an array of buffers to record the number of points assigned 
        // to each local component for each data group.
        val localComponentSize = Array.fill(numGroups)(Array.fill(1)(alpha).toBuffer)
        // Create the number of local components for each data group.
        val numLocalComponents = Array.fill(numGroups)(0).toBuffer

        // Create the number of global components.
        var numComponents = 1

        // Create an extensible matrix to hold the global labels that each 
        // local label is associated with.  This will have numGroups labels
        // and the number of columns will vary and mutate based on the data 
        // group.
        var globalLabels = Array.fill(numGroups)(Array.fill(0)(0).toBuffer)

        // Create a buffer to hold statistics describing each global component:
        //   1) number of points assigned to each component 
        //   2) the likelihood of selecting this component
        //   3) a cummulative vector for the component
        //   4) a cummulative error for the component.
        // You can build the mean and covariance by diving the cumulative 
        // vector and error vectors by the number of points from each 
        // component.   We keep them separate so that it's easier to remove a
        // data point from the cumulative vectors.
        val globalComponents = Array(computeInitialComponent(data, numPoints)).toBuffer

        // Perform a number of global iterations over all data groups.
        for (n <- 0 until numIter) {
            printf("Starting Iteration [%d] with [%d] components \n", n, globalComponents.size-1)
            // Traverse each data group.
            for ( (group_j, j) <- data.zipWithIndex) {
                var localComponentCount = numLocalComponents(j)
                printf("Starting data group [%d] with [%d] components\n", j, localComponentCount)
                // Traverse each data point in a single group.
                for ( (x_ji, i) <- group_j.zipWithIndex ) {
                    def likelihood(theta: Theta) = gaussian(x_ji, theta._2, theta._3)

                    val t_ji = localLabels(j)(i)
                    if (n != 0) {
                        val k_jt = globalLabels(j)(t_ji-1)
                        globalComponents(k_jt) = updateComponent(globalComponents(k_jt), -1)
                        localComponentSize(j)(t_ji) -= 1
                    } 

                    // Compute the likelihood of each global component creating
                    // this data point.
                    val fullLikelihood = DenseVectorRow[Double](
                        globalComponents.map(likelihood).toArray)

                    // Compute the prior on the global set of components.  
                    // This uses the second element in the component tuples.
                    val globalPrior = norm(DenseVectorRow[Double](
                        globalComponents.map(_._1).toArray))
                    // Compute the posterior for each of the global components 
                    // for this data point.
                    val globalPosterior = fullLikelihood
                    // Compute the full probability of each global component.
                    val globalProbs = globalPrior :* globalPosterior

                    // Accumulate this to get the likelihood of selecting a new 
                    // global component rather than a local component for this 
                    // data point.
                    val newComponentLikelihood = globalProbs.sum

                    // Pull out the likelihood of each local component
                    // (which are really just global components) for this 
                    // data group.
                    val localLikelihood = globalLabels(j).map(fullLikelihood)

                    // Compute the prior for each local component.
                    val prior = DenseVectorRow[Double](localComponentSize(j).toArray)
                    // Compute the posterior for each local component.
                    val posterior = DenseVectorRow[Double]((
                        newComponentLikelihood +: localLikelihood).toArray)
                    // Combine the two to get the full likelihood of each component.
                    val probs = norm(prior :* posterior)

                    // Sample a new local label for this data point.
                    val t_ji_new = new Multinomial(probs).sample

                    if (t_ji_new == 0) {
                        // If we selected a new local component, we have to 
                        // sample to see if the component links to an existing 
                        // global component or a new one.

                        // First, update the number of local components .
                        localComponentCount += 1

                        // Now sample from the normalized likelihood of each 
                        // global component.
                        val k_jt_new = new Multinomial(norm(globalPrior)).sample

                        // Assign the data point a brand new id for the new 
                        // local label.  This will just be the next new 
                        // available label for this data group.
                        localLabels(j)(i) = globalLabels(j).size + 1
                        // Add a new size for the new local component.
                        localComponentSize(j).append(1)

                        if (k_jt_new == 0) {
                            // If we generated a new global component, append 
                            // it's new index to the mapping from local 
                            // component id's to global component id's and 
                            // create the new component.
                            globalLabels(j).append(globalComponents.size)
                            globalComponents.append(newComponent(x_ji))
                        } else {
                            // Otherwise just update the labeling using an 
                            // existing global component id.
                            globalLabels(j).append(k_jt_new)
                            globalComponents(k_jt_new) = updateComponent(globalComponents(k_jt_new), 1)
                        }
                    } else {
                        // If we selected an existing local label, just update this in the local labelings.
                        localLabels(j)(i) = t_ji_new
                        localComponentSize(j)(t_ji_new) += 1
                        val k_tj_new = globalLabels(j)(t_ji_new-1)
                        globalComponents(k_tj_new) = updateComponent(globalComponents(k_tj_new), 1)
                    }
                }

                // After processing each data point within the local group, 
                // formulate the local means and sample a new global mean for 
                // each one.
                numLocalComponents(j) = localComponentCount
                val localComponents = localLabels(j).zip(group_j)
                                                    .groupBy(_._1)
                                                    .toList
                                                    .map(formLocalCenter)
                val newLocalLabels = for ( ((n_t, mu_t, t), t_new) <- localComponents.zipWithIndex ) yield {
                    val k_jt = globalLabels(j)(t)
                    globalComponents(k_jt) = updateComponent(globalComponents(k_jt), -n_t)
                    def likelihood(theta: Theta) = gaussian(mu_t, theta._2, theta._3)
                    val prior = norm(DenseVectorRow[Double](
                        globalComponents.map(_._1).toArray))
                    val posterior = DenseVectorRow[Double](
                        globalComponents.map(likelihood).toArray)
                    val probs = norm(prior :* posterior)
                    val k_jt_new = new Multinomial(probs).sample
                    if (k_jt_new == 0)
                        globalComponents.append(newComponent(mu_t))
                    else
                        globalComponents(k_jt_new) = updateComponent(globalComponents(k_jt_new), n_t)
                    localComponentSize(j)(t_new+1) = n_t

                    (if (k_jt_new == 0) globalComponents.size-1 else k_jt_new, (t+1, t_new+1))
                }
                localComponentSize(j).trimEnd(localComponentSize(j).size - newLocalLabels.size - 1)
                globalLabels(j) = newLocalLabels.map(_._1).toBuffer
                val remap = newLocalLabels.map(_._2).toMap
                localLabels(j) = localLabels(j).map(remap)
            }
            val labelRemap = (localLabels, globalLabels, data).zipped.toList.map{
                case (local, global, group) => local.map(t=>global(t-1)).zip(group)
            }.reduce(_++_)
             .groupBy(_._1)
             .map{ case (k,v) => (k, v.map(_._2)) }
             .zipWithIndex
             .map{ case ((k_old, x), k) => {
                 val n_k = x.size
                 val mu_k = x.reduce(_+_).toDense / n_k
                 val sigma_k = x.map(x_j => (mu_k - x_j) :^2).reduce(_+_)/n_k + epsilon
                 globalComponents(k+1) = (n_k, mu_k, sigma_k)
                 (k_old, k+1)
            }}
            globalComponents.trimEnd(globalComponents.size - (labelRemap.size+1))
            globalComponents.foreach(theta => println(theta._2))
            globalLabels = globalLabels.map(global => global.map(labelRemap))
        }

        for ( (local, global) <- localLabels.zip(globalLabels) ) yield
            local.map(t=>global(t-1))
    }

    def computeInitialComponent(data: List[List[VectorRow[Double]]],
                                numPoints: Int) =  {
        val mu = data.map(_.reduce(_+_)).reduce(_+_).toDense / numPoints
        val sigma = data.reduce(_++_)
                        .map(x_i => (mu - x_i) :^ 2)
                        .reduce(_+_) / numPoints
        (gamma, mu, sigma + epsilon)
    }

    def newComponent(x: VectorRow[Double]) = 
        (1d, x.toDense, DenseVectorRow.ones[Double](x.length))
    def updateComponent(theta: Theta, delta: Int) =
        (theta._1+delta, theta._2, theta._3)
    def formLocalCenter( entry: (Int, Array[(Int, VectorRow[Double])])) = {
        val t = entry._1
        val n_t = entry._2.size
        val mu_t = entry._2.map(_._2).reduce(_+_).toDense/n_t
        (n_t, mu_t, t-1)
    }
}
