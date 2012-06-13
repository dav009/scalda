import AssemblyKeys._ // put this at the top of the file

seq(assemblySettings: _*)

name := "OnlineLDA"

version := "0.1"

scalaVersion := "2.9.1"

mainClass in assembly := Some("edu.ucla.sspace.OnlineTopicModel")

resolvers += "Local Maven Repository" at "file://"+Path.userHome.absolutePath+"/.m2/repository"

libraryDependencies += "org.scalala" %% "scalala" % "1.0.0.RC3-SNAPSHOT"

libraryDependencies += "org.scalanlp" %% "scalanlp-learn" % "0.4.RC1"

libraryDependencies += "edu.ucla.sspace" % "sspace-wordsi" % "2.0"

libraryDependencies += "com.googlecode.netlib-java" % "netlib-java" % "0.9.3"

scalacOptions += "-deprecation"
