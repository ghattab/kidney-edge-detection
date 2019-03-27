library(ggplot2)
library(reshape2)

foo=read.table('~/Desktop/test.16.iouN.csv', header=TRUE, sep=';')
summary(foo)



default_aes = aes(
    shape = 19, colour = "gray", size = 1.5, fill = NA,
    alpha = NA, stroke = 0.5
  )



pdf("~/Desktop/test16.pdf")
foo$IOU.box = NULL # drop col
d <- melt(foo, id.vars="Frame")

# Everything on the same plot
p=ggplot(d, aes(Frame,value, col=variable)) + 
#  geom_line(size = 0.5)
  stat_smooth() 

# p=p + scale_color_brewer(palette="Dark2")+
  theme_minimal()
p + scale_color_manual(values=c("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")) + theme_minimal() + theme(legend.position="top") # bottom or none

dev.off()




pdf("~/Desktop/test16sep.pdf")
d <- melt(foo, id.vars="Frame")

# Separate plots
p = ggplot(d, aes(Frame,value)) + 
  geom_point(colour="darkgray", size = 1, stroke = 0, shape = 16) + 
  stat_smooth(colour="black", size=1) +
  facet_wrap(~variable)


p <- p + scale_color_brewer(palette="Paired")+
  theme_minimal()
# p + scale_color_grey() + theme_classic()
p

dev.off()
