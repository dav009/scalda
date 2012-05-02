package edu.ucla.sspace.learner

import edu.ucla.sspace.learner.gibbs.LDA

import scalala.tensor._

import scala.io.Source


object BatchTopicModel {
    def main(args: Array[String]) {
        val topicModel = new LDA(
            Source.fromFile(args(0)).getLines.toSet, // Stop Words
            1000, // Num iterations
            0.01, 0.01) // Alpha and Beta
        topicModel.train(args(1), args(2).toInt)
    }
}
