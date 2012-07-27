library(ggplot2)

plotGrouping <- function(baseName) {
data <- data.frame(read.table(paste(baseName, "txt", sep="."), header=TRUE))
p <- ggplot(data, aes(x=X, y=Y)) +
     geom_point(aes(colour=factor(Group))) +
     facet_grid(. ~ Iteration) +
     opts(legend.position = "none")
ggsave(paste(baseName, "png", sep="."), width=7*4, height=7)
}

#plotGrouping("vdpmm")
#plotGrouping("test.hdpmm.groups")
plotGrouping("test.isgmm.groups")
#plotGrouping("test.immm.groups")
#plotGrouping("test.gdpmm.groups")
#plotGrouping("test.nb.groups")
#plotGrouping("test.nb.3.groups")
#plotGrouping("test.vmf.3.groups")
#plotGrouping("test.dpvmf.3.groups")
#plotGrouping("test.sgmm.groups")
#plotGrouping("test.gmm.groups")
#plotGrouping("test.km.groups")

plotMap <- function(baseName) {
base_size <- 10
data <- data.frame(read.table(paste(baseName, "txt", sep="."), header=TRUE))
data <- ddply(data, .(Category, Group), nrow)
data <- ddply(data, .(Category), transform, rescale = rescale(V1))
p <- ggplot(data, aes(x=factor(Category), y=factor(Group))) +
     geom_tile(aes(fill = rescale), colour = "white") +
     scale_fill_gradient(low = "white", high = "steelblue") +
     theme_bw(base_size = base_size) + 
     labs(x = "", y = "") + 
     scale_x_discrete(expand = c(0, 0)) +
     opts(legend.position = "none",
          axis.ticks = theme_blank(),
          axis.text.x = theme_text(size = base_size * 0.8, 
                                   angle = 280, 
                                   hjust = 0.5,
                                   colour = "grey50"))

ggsave(paste(baseName, "png", sep="."))
}

#plotMap("mdconalds.gmm.groups")
#plotMap("mdconalds.km.groups")
#plotMap("mdconalds.gdpmm.groups")
plotMap("mdconalds.nb.groups")
