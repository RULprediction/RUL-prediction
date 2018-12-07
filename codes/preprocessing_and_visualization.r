########### packages ##########
library(reshape)
library(scatterplot3d)
library(e1071)
library(cluster)
library(MASS)
#library(VisuClust)
#library(devtools)
library(ProjectionBasedClustering)
library(ggplot2)
library(RSNNS)
library(flexclust)
library(caret)
library(Boruta)
library(randomForest)
################# read and rename datasets ###########
### dataset002: train set has 260 units and test set has 259 units
### has 1 fault mode and 6 operational conditions
dat2<-read.table("C:/train2.txt",header = F)
dat2<-rename(dat2,c("V1"="unit","V2"="cycle","V3"="setting1","V4"="setting2","V5"="setting3","V6"="feature1","V7"="feature2","V8"="feature3","V9"="feature4","V10"="feature5","V11"="feature6","V12"="feature7","V13"="feature8","V14"="feature9","V15"="feature10","V16"="feature11","V17"="feature12","V18"="feature13","V19"="feature14","V20"="feature15","V21"="feature16","V22"="feature17","V23"="feature18","V24"="feature19","V25"="feature20","V26"="feature21"))
test2<-read.table("C:/test2.txt",header = F)
test2<-rename(test2,c("V1"="unit","V2"="cycle","V3"="setting1","V4"="setting2","V5"="setting3","V6"="feature1","V7"="feature2","V8"="feature3","V9"="feature4","V10"="feature5","V11"="feature6","V12"="feature7","V13"="feature8","V14"="feature9","V15"="feature10","V16"="feature11","V17"="feature12","V18"="feature13","V19"="feature14","V20"="feature15","V21"="feature16","V22"="feature17","V23"="feature18","V24"="feature19","V25"="feature20","V26"="feature21"))
RUL2<-read.table("C:/RUL2.txt",header = F) # True RUL for the test dataset

######################################################
######### define the RUL for train dataset##########
###dataset002
RUL=rep(0,nrow(dat2))
k=1
for(i in 1:260){
  
  n=nrow(dat2[which(dat2[,1]==i),])
  m=k+n-1
  for(j in k:m){
    RUL[j]=n-dat2[j,2]
    
  } 
  k=k+n
}
dat2=cbind(RUL,dat2)

#################  3D plot for the three operational setting variables ###########
scatterplot3d(dat2[,4],dat2[,5],dat2[,6],
              pch=16,
              highlight.3d = TRUE,
              main = "3D Scatter Plot of three setting variables from dataset002",
              xlab="setting 1",ylab="setting 2",zlab="setting 3")

########### using kmeans to check the cluster result ###########
setting=dat2[,c(4,5,6)]
km = kcca(setting,k=6,kccaFamily("kmeans"))
cls=predict(km)
dat2_new=cbind(cls,dat2) # adding the class varaible of different operational setting


clus=cbind(setting,cls)
par(mfrow=c(1,1))
scatterplot3d(clus[,1],clus[,2],clus[,3],
              pch=16,
              color=clus[,4],
              main = "3D Scatter Plot of three setting variables from dataset002",
              xlab="setting 1",ylab="setting 2",zlab="setting 3")
################################################################################################
#########  Topographic transformation of data using Sammon mapping  ###########################

### get a sample dataset(because the number of whole instances is so large)
sam=sample(1:nrow(dat2), 8000, replace = FALSE)
D=dist(dat2[sam,])
S=sammon(D) # MDS for visualization
par(mfrow=c(1,1))
SS=as.matrix(S$points)
SS=cbind(SS,dat2[sam,1])
SS=as.data.frame(SS)

SS<-rename(SS,c("V3"="RUL"))
# visualization
ggplot(SS, aes(x = V1, y = V2, colour = RUL))+geom_point(alpha = .5)
plot(S$points,color=dat2[sam,1])

################################################################################################
#############  Normalization under different condition modes  ###########################
dat2_new2=dat2_new # dat2_new2 is normalized dataset
for(i in 1:6){
  for(j in 1:24){
    dat2_new2[which(dat2_new[,1]==i),j+4]=normalizeData(dat2_new[which(dat2_new[,1]==i),j+4])
  }
}  # Normalization for both setting variables and sensor data

############################ add 6 variables containing modes history#########################
mode1=rep(0,nrow(dat2_new)) # counting the number of No.1 mode in the whole life for each unit 
mode2=rep(0,nrow(dat2_new))
mode3=rep(0,nrow(dat2_new))
mode4=rep(0,nrow(dat2_new))
mode5=rep(0,nrow(dat2_new))
mode6=rep(0,nrow(dat2_new))
dat2_new3=cbind(mode1,mode2,mode3,mode4,mode5,mode6,dat2_new2)

m=0
for(k in 1:260){
  
  dk=dat2_new3[which(dat2_new3[,9]==k),]
  for(i in 1:6)
  {
    for(j in 1:nrow(dk))
    {
      dat2_new3[m+j,i]=length(which(dk[c(1:j),7]==i))
    }
    
  }
  m=m+nrow(dk)
}
dat2_new4=cbind(dat2_new3[,1:6],dat2_new)
# new dataset with all labels and after normalization
write.csv(dat2_new3,file="C:/Dataset 002_train data after normalization.csv")
# new dataset with all labels and without normalization
write.csv(dat2_new4,file="C:/Dataset 002_train data without normalization.csv")


########### same format for test dataset(259 units) ############
### True RUL
RUL=rep(0,nrow(test2))
k=1
for(i in 1:259){
  
  n=nrow(test2[which(test2[,1]==i),])
  m=k+n-1
  rul=RUL2[i,]
  for(j in k:m){
    RUL[j]=n-test2[j,2]+rul
    
  } 
  k=k+n
}
test2=cbind(RUL,test2)
### using the same classifier to predict the test dataset
setting_test=test2[,c(4,5,6)]
pred=predict(km,newdata=setting_test)
cls=pred
tt=cbind(cls,test2)
tt_new=tt
for(i in 1:6){
  for(j in 1:24){
    tt_new[which(tt_new[,1]==i),j+4]=normalizeData(tt_new[which(tt_new[,1]==i),j+4])
  }
} # tt_new is the normalized under different modes

#### Adding 6 variables
mode1=rep(0,nrow(tt))
mode2=rep(0,nrow(tt))
mode3=rep(0,nrow(tt))
mode4=rep(0,nrow(tt))
mode5=rep(0,nrow(tt))
mode6=rep(0,nrow(tt))
tt1=cbind(mode1,mode2,mode3,mode4,mode5,mode6,tt_new)

m=0
for(k in 1:259){
  
  dk=tt1[which(tt1[,9]==k),]
  for(i in 1:6)
  {
    for(j in 1:nrow(dk))
    {
      tt1[m+j,i]=length(which(dk[c(1:j),7]==i))
    }
    
  }
  m=m+nrow(dk)
}
# new dataset with all labels and after normalization
write.csv(tt1,file="C:/Dataset 002_test data after normalization.csv")
# new dataset with all labels and without normalization
tt1_1=cbind(tt1[,1:6],tt)
write.csv(tt1_1,file="C:/Dataset 002_test data without normalization.csv")




