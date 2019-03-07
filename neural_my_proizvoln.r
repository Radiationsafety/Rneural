#по мотивам https://www.youtube.com/watch?v=C_AWVOqTD1I - определение цены вина
#install.packages('neuralnet')
library(neuralnet)
# Data preparation
getwd()
data <- read.csv("winemag-data_first150k.csv", header = TRUE) # считает минут 10 для 150 000 значений, для 10000 - менее 30 сек.
str(data)
data<-data[,c(2,5,6,7,10)]
data<-data[which(is.na(data$price)==FALSE),]

#from text to digit 0-1
x1 <- as.factor(data$country)
levels(x1) <- 1:length(levels(x1))
x2 <- as.factor(data$province)
levels(x2) <- 1:length(levels(x2))
x3 <- as.factor(data$variety)
levels(x3) <- 1:length(levels(x3))

data2<-as.data.frame(cbind(x1,data$points,data$price,x2,x3))
names(data2)=c("country","points","price","province","variety")
data3<-data2
# Min-Max Normalization
data2$country <- (data2$country - min(data2$country))/(max(data2$country) - min(data2$country))
data2$points <- (data2$points - min(data2$points))/(max(data2$points) - min(data2$points))
data2$price <- (data2$price - min(data2$price))/(max(data2$price) - min(data2$price))
data2$province <- (data2$province - min(data2$province))/(max(data2$province) - min(data2$province))
data2$variety <- (data2$variety - min(data2$variety))/(max(data2$variety) - min(data2$variety))

# Data Partition
set.seed(222)
ind <- sample(2, nrow(data2), replace = TRUE, prob = c(0.7, 0.3))
training <- data2[ind==1,]
testing <- data2[ind==2,]

# Neural Networks
set.seed(333)
n <- neuralnet(price~country+points+province+variety,
               data = training,
               hidden = 18,
               linear.output = FALSE)
plot(n)

# Prediction
#output <- compute(n, training[,-3])
#head(output$net.result)
#head(training[1,])

# Node Output Calculations with Sigmoid Activation Function
#in4 <- 0.0455 + (0.82344*0.7586206897) + (1.35186*0.8103448276) + (-0.87435*0.6666666667)
#out4 <- 1/(1+exp(-in4))
#in5 <- -7.06125 +(8.5741*out4)
#out5 <- 1/(1+exp(-in5))

#-------- предсказание стоимости вина по стране, очкам,провинции и его типу
i=7
#input<-training[i,-3]

# расчёт для произвольных значений, от изменения рейтинга менятеся стоимость вина!
input<-as.vector(c(0.9791667,0.60,0.1142857,0.6386688))
input<-rbind(input,c(0,0,0,0))
names(input)<-c("country","points","province","variety")
input<-as.data.frame(input)
output <- compute(n, input[1,])
res1 = output$net.result * ( max(data3$price) - min(data3$price) ) + min(data3$price)
print(paste("net result = ",
            format(res1,nsmall=1),
            "$",sep=""))

#---------------------------------


# Confusion Matrix & Misclassification Error - training data
output <- compute(n, training[,-3])
p1 <- output$net.result
pred1 <- ifelse(p1>0.5, 1, 0)
tab1 <- table(pred1, training$price)
tab1
1-sum(diag(tab1))/sum(tab1)

# Confusion Matrix & Misclassification Error - testing data
output <- compute(n, testing[,-3])
p2 <- output$net.result
pred2 <- ifelse(p2>0.5, 1, 0)
tab2 <- table(pred2, testing$price)
tab2
1-sum(diag(tab2))/sum(tab2)
