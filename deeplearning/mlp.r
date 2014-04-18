library(monmlp)
test1 = read.csv("C:\\...\\iono.csv")
x = as.matrix(test1[,1:34])
y = as.matrix(test1[,35])
w.mon <- monmlp.fit(x = x, y = y, hidden1 = 3, monotone = 1,n.ensemble = 15, bag = TRUE)
p.mon <- monmlp.predict(x = x, weights = w.mon)
p.mon.ma = matrix(unlist(attr(p.mon, "ensemble")),ncol = 15)
p.mon.mean = apply(p.mon.ma,1, mean)
table(p.mon.mean[p.mon.mean>0.4],y)
#        0   1
#  FALSE 223  11
#  TRUE    1 115
