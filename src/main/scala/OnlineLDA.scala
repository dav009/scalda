package edu.ucla.sspace

import scalala.collection.sparse.SparseArray
import scalala.library.Library._
import scalala.library.Library.Axis._
import scalala.library.Numerics._
import scalala.tensor._
import scalala.tensor.dense._
import scalanlp.stats.distributions.Gamma

class OnlineLDA(val numWords: Int,
                val numTopics: Int, 
                val numDocs: Int) {
    val gammaDist = new Gamma(100.0, 1.0/100.0)
    val alpha = 1.0/numTopics
    val eta = 1.0/numTopics
    val tau = 1021
    val kappa = 0.7
    var epoch = 0

    val lambda = newGamma(numTopics, numWords)
    var beta = expectation(lambda).mapValues(exp(_))

    def processDocument(doc: SparseArray[Double]) {
        // Compute the sufficient statistics needed to update lambda.
        val sstats = computeExpectation(doc)
        // Compute the decay rate based on the current training epoc.
        val rhot = pow(tau+epoch, -kappa)
        // Update lambda.
        lambda.transformTriples( 
            (r,c,v) => v * (1-rhot) + rhot * (eta + numDocs * sstats(r,c)))
        beta = expectation(lambda).mapValues(exp(_))
        epoch += 1
    }

    def computeExpectation(doc: SparseArray[Double]) = {
        // Extract the indices and word counts for the document.
        val ids = doc.indexArray
        val counts = new DenseMatrix[Double](1, ids.size, doc.valueArray)

        // Extract the topic probabilities for each word we are concerned with.
        // This will be of size
        //   numTopics X numWordsInDoc
        val beta_d = beta(::, ids)

        def update(docGamma: DenseMatrix[Double]) = {
            // Extract the topic probabilities for each word we are concerned with.
            // The theta value is based on a prior gamma matrix that will be
            // iteraively updated.  The sizes will be:
            //   1 X numTopics
            var theta = expectation(docGamma).mapValues(exp(_))
            // Compute the phi matrix used to modify gamma.  This will be sized
            //   1 X numWordsInDoc
            var phiNorm = theta * beta_d + 1e-100
            (theta, phiNorm)
        }

        // We recompute gamma, matrices will have dimensions:
        //   alpha: scalar
        //   docTheta: 1 X numTopics
        //   counts, phiNorm: 1 X numWordsInDoc
        //   beta_d: numTopics X numWordsInDoc
        // So counts :/ phiNorm -> 1 X numWordsInDoc
        //    above * beta_d.t -> 1 X numTopics
        //    docTheta :* above -> 1 X numTopics
        // as desired.
        val (theta, phi) = (0 until 1).foldLeft(update(newGamma(1, numTopics))){
            (params, _) => {
                val (theta, phi) = params
                update(theta :* ((counts :/ phi) * beta_d.t))
            }}
        // We update the sufficient statistics to update lamda by utilizing
        // theta and the normalized phi matrix.
        var sstats = DenseMatrix.zeros[Double](lambda.numRows, lambda.numCols)
        sstats(::, ids) :+= theta.t * (counts :/ phi)
        // The final update is of size
        //   numTopics X numWords * numTopics * numWords
        sstats :* beta
    }

    def newGamma(rows: Int, columns: Int) =
        new DenseMatrix[Double](rows, columns,
                                gammaDist.sample(rows*columns).toArray)

    def expectation(values: DenseMatrix[Double]) = {
        val docExpectations = sum(values, Vertical).mapValues(digamma(_))
        values.mapTriples( (r, c, v) => digamma(v) - docExpectations(r))
    }
}
