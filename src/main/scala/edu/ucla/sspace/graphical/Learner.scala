package edu.ucla.sspace.graphical

import breeze.linalg.Vector

import java.io.PrintWriter


trait Learner {
   def train(data: List[Vector[Double]], 
             numGroups: Int,
             priorData: List[List[Vector[Double]]] = List(List[Vector[Double]]())) : Array[Int]

   var dataPrinter: () => List[String] = null;
   var printer: PrintWriter = null;
   def setReporter(p: PrintWriter, d: () => List[String] ) { 
       printer = p
       dataPrinter = d
   }

   def report(iteration: Int, labels: List[Int]) { 
       for ( (d, l) <- dataPrinter().zip(labels) )
           printer.println("%s %d %d".format(d, iteration, l))
       printer.flush
   }
}
