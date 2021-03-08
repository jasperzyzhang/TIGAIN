#Data Summary and Visulization for TIGIAN resutls
# Jasper Zhang

#load library needed
library(Rmisc)
library(gmodels)
library(ggplot2)
library(readr)

#specify decimal for data x
specify_decimal <- function(x, k) trimws(format(round(x, k), nsmall=k))

#summarized the mean and standard error of 10 experiments
se_sum <- function(data){
  format_gp(group.STDERR(RMSE ~ type +colnum, data = data))
}
format_gp <- function(data){
  data$SE = specify_decimal(data$RMSE.upper - data$RMSE.mean,4)
  print(data,digits = 3)
}

se_sum <- function(data){
  format_gp(group.STDERR(RMSE ~ type +colnum, data = data))
}
#resultpath
rpath = '/Users/zhongyuanzhang/Dropbox/2021Winter/task/GAIN-master/result/'

#read result file
iGE= read.csv(paste(rpath,"ImputeGE.csv",sep="") ,header = TRUE)
iCN= read.csv(paste(rpath,"ImputeCN.csv",sep="") ,header = TRUE)
iME= read.csv(paste(rpath,"ImputeME.csv",sep="") ,header = TRUE)

#summarize results
se_sum(iGE)
se_sum(iCN)
se_sum(iME)