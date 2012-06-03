package edu.ucla.sspace.graphical

import edu.ucla.sspace.graphical.gibbs.HierarchicalDirichletProcessMixtureModel

import scalala.tensor.dense.DenseVector

import scala.io.Source

import java.io.PrintWriter


object RunHLearner {
    def main(args:Array[String]) {
        if (args.size < 4) {
            printf("usage: RunLearner <learnerType> <numClusters> <outFile> <data.mat>+\n")
            System.exit(1)
        }

        val data = args.view(3, args.size).map(loadFile).toList

        val nTrials = 100
        val k = args(1).toInt
        val learner = args(0) match {
            case "hdpmm" => new HierarchicalDirichletProcessMixtureModel(1d, 1d, nTrials)
        }

        val assignments = learner.train(data.toList)
        val w = new PrintWriter(args(2))
        w.println("X Y Group")
        for ( (group_j, labels_j) <- data.zip(assignments);
              (x_ji, l_ji) <- group_j.zip(labels_j) )
            w.println("%f %f %d".format(x_ji(0), x_ji(1), l_ji))
        w.close
    }

    def convertLine(line: String) =
        DenseVector[Double](line.split("\\s+").map(_.toDouble)).t

    def loadFile(file: String) =
        Source.fromFile(file).getLines.map(convertLine).toList
}
