package edu.ucla.sspace

import scalala.collection.sparse.SparseArray
import scalala.tensor._

import edu.ucla.sspace.basis.StringBasisMapping
import edu.ucla.sspace.text.OneLinePerDocumentIterator

import scala.collection.JavaConversions.asScalaIterator
import scala.collection.JavaConversions.setAsJavaSet
import scala.io.Source._

import java.util.HashSet


object OnlineTopicModel {
    def main(args: Array[String]) {
        val words = new HashSet[String]()
        fromFile(args(0)).getLines.foreach(words.add(_))
        val basis = new StringBasisMapping(words)
        basis.setReadOnly(true)
        val numTopics = args(1).toInt
        val numDocs = args(2).toInt
        val learner = new OnlineLDA(basis.numDimensions, numTopics, numDocs)

        for ((doc,i) <- new OneLinePerDocumentIterator(args(3)).zipWithIndex) {
            printf("Processing document %d\n", i)
            learner.processDocument(parseDoc(doc.reader.readLine))
            if (i % 20 == 0) {
                val beta = learner.beta
                for (r <- 0 until beta.numRows) 
                    println(beta(r,::).toMap.toList
                                      .map(e => (e._2, e._1))
                                      .sorted.reverse
                                      .map(e => basis.getDimensionDescription(e._2))
                                      .take(10).mkString(" "))
            }
        }
        def parseDoc(text: String) = {
            val vec = new SparseArray[Double](basis.numDimensions)
            text.split("\\s+").map(basis.getDimension(_))
                              .filter(_ >= 0)
                              .foreach(vec.update(_, 1d))
            vec
        }
    }
}
