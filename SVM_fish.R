 

fish<-read.csv("~/SVM-practice/Fishdata.csv", head = T) 
library(e1071) 
s<-svm(fish[,2:7],fish[,1]) 
summary(s) 
# 分不太好
pred<-predict(s,fish[,2:7])
table(fish[,1],pred)
temp <- table(fish[,1],pred)
# The apparent error rate
1 - sum(diag(temp))/length(fish[,1])
# The apparent error rate is 25/148.
# We can change gamma to get a new (or better) classifier:
s1<-svm(fish[,2:7],fish[,1],gamma=1)
pred1<-predict(s1,fish[,2:7])
table(fish[,1],pred1)
temp <- table(fish[,1],pred1)
# The apparent error rate
1 - sum(diag(temp))/length(fish[,1])
# The new error rate is 16/148, which becomes significantly smaller.

#Choose an even larger gamma:
s5<-svm(fish[,2:7],fish[,1],gamma=5)
pred5<-predict(s5,fish[,2:7])
table(fish[,1],pred5)
temp <- table(fish[,1],pred5)
# The apparent error rate
1 - sum(diag(temp))/length(fish[,1])
# The new error rate is 9/148.

# gamma = 1000
#Choose an even larger gamma:
s1000<-svm(fish[,2:7],fish[,1],gamma=1000)
pred1000<-predict(s1000,fish[,2:7])
table(fish[,1],pred1000)
temp <- table(fish[,1],pred1000)
# The apparent error rate
1 - sum(diag(temp))/length(fish[,1])

# Theoretically, the increase of gamma will derive an apparent error rate 0.
# However, this might cause an over-fitting problem which affects the "true error rate".
# Now we use a 20-fold CV to derive a good combination of (cost,gamma) based on grid search:
    
# (1) Let's start with (cost=0.1,gamma=0.1):
c1<-svm(fish[,2:7],fish[,1],cost=0.1,gamma=0.1,cross=20) # cross -> cross validation
summary(c1)
# ..........................................
# 20-fold cross-validation on training data:
  
# Total Accuracy: 62.83784
  
# (2) Change to (cost=0.5,gamma=0.1):
c1<-svm(fish[,2:7],fish[,1],cost=0.5,gamma=0.1,cross=20)
summary(c1)

# ..........................................
# 20-fold cross-validation on training data:
  
#Total Accuracy: 82.43243
  
#(3) Change to (cost=100,gamma=0.2): 
c1<-svm(fish[,2:7],fish[,1],cost=130,gamma=0.2,cross=20)
summary(c1)
# ..........................................
# 20-fold cross-validation on training data:
  
# Total Accuracy: 90.54054
  
# Question: How to find the best combination of (cost, gamma) that minimizes the prediction error?
  
#  To find the best combination of the tuning parameters (cost, gamma), one can perform a grid search 
#  over a respecified parameter range. Suppose now we search the best (cost, gamma) over the region 
#  of [100,1000]x[0.5,5], with 10 equally spaced points allocated for each dimension (thus 100 grid points to be compared).
#  The result based on a 10-fold cross validation (default) can be produced by:
  
tobj <- tune.svm(Species ~ ., data=fish, cost= 100*(1:10), gamma=0.5*(1:10))
summary(tobj)
  
# Parameter tuning of ‘svm’:
  
# - sampling method: 10-fold cross validation 
  
# - best parameters:
# gamma cost
# 0.5  300
  
# - best performance: 0.06714286 <- true error rate
  
# - Detailed performance results:
#  gamma cost      error dispersion
#  1     0.5  100 0.08761905 0.05585858
#  2     1.0  100 0.08666667 0.07730012
#  3     1.5  100 0.10714286 0.07799613
#  4     2.0  100 0.14095238 0.09582360
#  5     2.5  100 0.16095238 0.10888472 
#  .
#  .
#  .
#  98    4.0 1000 0.20857143 0.11021864
#  99    4.5 1000 0.22285714 0.10473304
#  100   5.0 1000 0.23000000 0.11170025
#  
#  This shows the best choice of (cost, gamma) is (300, 0.5), which results in a minimum prediction error 0.067
  
#  For the overall comparison, one can produce the contour plot of prediction errors over the search range of (cost, gamma) by:
  


plot(tobj, xlab = "gamma", ylab="C")
  
  
# Based on the optimal choice (cost=300, gamma=0.5), we can fit a new SVM model for all training data:
c1<-svm(fish[,2:7],fish[,1],cost=300,gamma=0.5)
pred<-predict(c1,fish[,2:7])
table(pred,fish[,1])
temp <- table(pred,fish[,1])
# The apparent error rate
1 - sum(diag(temp))/length(fish[,1])
# The apparent error rate 為0

  
#  pred    bream parki perch pike roach smelt white
#  bream    33     0     0    0     0     0     0
#  parki     0    10     0    0     0     0     0
#  perch     0     0    54    0     0     0     0
#  pike      0     0     0   16     0     0     0
#  roach     0     0     0    0    18     0     0
#  smelt     0     0     0    0     0    12     0
#  white     0     0     0    0     0     0     5
  
#  ==> Note that this classifier has a “zero apparent error rate”.
#  The most important thing is, it is a better classifier for prediction (the estimated true error is 0.067).
#  Now we use this classifier to predict the test data:

test<-read.csv("~/SVM-practice/Fish_test_data.csv")
predict(c1,test)
#  1     2     3     4     5     6     7     8     9    10    11 
#  parki bream perch perch  pike smelt smelt parki roach roach perch 
#  Levels: bream parki perch pike roach smelt white
#  
#  Check out if the prediction results the same as those made by other approaches.
  
