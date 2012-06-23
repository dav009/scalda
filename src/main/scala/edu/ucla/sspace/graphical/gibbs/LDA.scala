package edu.ucla.sspace.learner.gibbs

import edu.ucla.sspace.basis.BasisMapping
import edu.ucla.sspace.basis.FilteredStringBasisMapping
import edu.ucla.sspace.matrix.Matrix
import edu.ucla.sspace.matrix.GrowingSparseMatrix
import edu.ucla.sspace.matrix.ArrayMatrix
import edu.ucla.sspace.text.OneLinePerDocumentIterator
import edu.ucla.sspace.text.Document
import edu.ucla.sspace.vector.DenseVector
import edu.ucla.sspace.vector.DoubleVector

import breeze.numerics._
import breeze.linalg._

import scala.collection.JavaConversions.asScalaIterator
import scala.collection.JavaConversions.setAsJavaSet
import scala.io.Source
import scala.util.Random._

import java.util.HashSet


class LDA(stopWords:Set[String], 
          val numIterations:Int,
          val alpha:Double,
          val beta:Double) {
    var docTopics:DenseMatrix[Double] = null
    var topicWords:DenseMatrix[Double] = null

    val basis = new FilteredStringBasisMapping(stopWords)

    def train(corpus:String, numTopics:Int) {
        val numDocs = getDocuments(corpus).map(d=>{getWords(d._1); 1}).sum
        val numWords = basis.numDimensions
        basis.setReadOnly(true)

        // Create the counts for the document by topic events.
        docTopics = DenseMatrix.zeros[Double](numDocs, numTopics) + alpha

        // Create the counts for the topic by word events.  This will become dense but
        // we don't know how many words we'll have until we're done
        topicWords = DenseMatrix.zeros[Double](numTopics, numWords) + beta

        // Create an array tracking the number of words assigned to each topic
        // across all documents.
        val topicCounts = new DenseVectorRow[Double](Array.ofDim[Double](numTopics)) +
                          (numWords * beta)

        // Create a matrix tracking the document by word topic assignments.  This will
        // just hold the topic id in each cell.
        val assignments = new GrowingSparseMatrix(numDocs, 0) 

        for (i <- 0 until numIterations+1) {
            printf("[Processing Iteration %d]\n", i)
            for ((doc, d) <- getDocuments(corpus); word <- getWords(doc)) {
                if (i == 0) {
                    val currTopic = assignments.get(d, word).toInt
                    docTopics(d, currTopic) -= 1
                    topicWords(currTopic, word) -= 1
                    topicCounts(currTopic) -= 1
                }

                val topicProbs = topicWords(::,word).t :* docTopics(d,::) :/ topicCounts
                val topic = sampleMultinomial(topicProbs)

                assignments.set(d, word, topic)
                docTopics(d, topic) += 1
                topicWords(topic, word) += 1
                topicCounts(topic) += 1

            }

            if (i % 20 == 0)
                printTopicWords
        }
    }

    def printDocTopics() {
        for (d <- 0 until docTopics.numRows)
            println(docTopics(d,::).data.zipWithIndex
                                   .sorted.reverse
                                   .map(_._2).mkString(" "))
    }

    def printTopicWords() {
        for (z <- 0 until topicWords.numRows)
            println(topicWords(z,::).toList.zipWithIndex
                                    .sorted.reverse.take(20)
                                    .map(_._2).map(i=>basis.getDimensionDescription(i))
                                    .mkString(" "))
    }

    def getDocuments(docFile: String) =
        (new OneLinePerDocumentIterator(docFile)).zipWithIndex

    def getWords(doc: Document) =
        doc.reader.readLine.toLowerCase.split("\\s+").map(basis.getDimension(_)).filter(_ >= 0)

    def sampleMultinomial(probabilities: DenseVectorRow[Double]) : Int = {
        val total = probabilities.sum
        var d = nextDouble
        for ((w, z) <- probabilities.data.zipWithIndex) {
            d -= (w/total)
            if (d <= 0)
                return z
        }
        return probabilities.size-1
    }
}
