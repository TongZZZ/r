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

test1 = read.table("C:\\...\\iono.csv",sep=",")
vars = names(test1)
data = as.data.frame(test1)
i <- sample(1:350, 200)
training <- data[i,]
test <- data[-i,]
training.x = training[,1:34]
training.y = training[,35]
test.x = test[,1:34]
test.y = test[,35]
f <- as.formula(paste("V35 ~", paste(vars[1:(length(vars)-1)], collapse="+")))
mylogit <- glm(f, data = training, family = "binomial")
#summary(mylogit)
prob=predict(mylogit,test.x ,type=c("response"))
table(prob>0.5, test.y)
#     test.y
#         0  1
#  FALSE 85 17
#  TRUE   8 41
