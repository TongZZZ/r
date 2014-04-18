library(monmlp)
test1 = read.csv("C:\\...\\iono.csv")
x = as.matrix(test1[,1:34])
y = as.matrix(test1[,35])
w.mon <- monmlp.fit(x = x, y = y, hidden1 = 3, monotone = 1,n.ensemble = 15, bag = TRUE)
p.mon <- monmlp.predict(x = x, weights = w.mon)
p.mon.ma = matrix(unlist(attr(p.mon, "ensemble")),ncol = 15)
p.mon.mean = apply(p.mon.ma,1, mean)
table(p.mon.mean>0.4,y)
#        0   1
#  FALSE 223  11
#  TRUE    1 115


data = as.matrix(test1)
i <- sample(1:350, 200)
training <- data[i,]
test <- data[-i,]
training.x = training[,1:34]
training.y = as.matrix(training[,35])
test.x = test[,1:34]
test.y = test[,35]

w.mon <- monmlp.fit(x = training.x, y = training.y, hidden1 = 3, monotone = 1,n.ensemble = 20, bag = TRUE)
p.mon <- monmlp.predict(x = test.x, weights = w.mon)
p.mon.ma = matrix(unlist(attr(p.mon, "ensemble")),ncol = 20)
p.mon.mean = apply(p.mon.ma,1, mean)
table(p.mon.mean>0.3,test.y)
#       test.y
#         0  1
#  FALSE 90 12
#  TRUE   3 45
