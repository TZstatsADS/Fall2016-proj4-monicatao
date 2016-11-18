source("http://bioconductor.org/biocLite.R")
biocLite("rhdf5")
library(rhdf5)

############################
#####song features#########
############################

dir<-"/Users/monicatao/Documents/ads/project4/Project4_data/data/"
files<-list.files(path=dir,"h5",recursive = T)


bar<-list()
beat<-list()
ld_max<-list()
ld_max_time<-list()
pitch<-list()
timbre<-list()
tatum<-list()


for(y in files){
    file=paste0(dir,y)
    sound<-h5read(file, "/analysis")
    name=strsplit(y,"/")[[1]][4]
    bar[[name]]<-sound$bars_start
    beat[[name]]<-sound$beats_start
    ld_max[[name]]<-sound$segments_loudness_max
    ld_max_time[[name]]<-sound$segments_loudness_max_time
    pitch[[name]]<-sound$segments_pitches
    timbre[[name]]<-sound$segments_timbre
    tatum[[name]]<-sound$tatums_start
  }

H5close()


feat<-list(bar,beat,ld_max,ld_max_time,tatum)
features<-matrix(nrow=0,ncol=length(feat)*7+12*21*2)
for(i in 1:2350){
  sum_feat<-vector()
  for(j in feat){
    sum_feat<-c(sum_feat,as.numeric(summary(j[[i]])),sd(j[[i]]))
  }
  
  samp_p<-1:ncol(pitch[[i]])
  c_p<-as.integer(quantile(samp_p,seq(0,1,0.05)))
  p<-as.vector(t(pitch[[i]][,c_p]))
  
  samp_t<-1:ncol(timbre[[i]])
  c_t<-as.integer(quantile(samp_t,seq(0,1,0.05)))
  t<-as.vector(t(timbre[[i]][,c_t]))
  
  sum_feat<-c(sum_feat,p,t)
  
  features<-rbind(features,sum_feat)
}
setwd("/Users/monicatao/Documents/ads/project4/Project4_data")
id<-read.table("common_id.txt")
rownames(features)<-id[,1]


save(bar,file="bar.RData")
save(beat,file="beat.RData")
save(ld_max,file="ld_max.RData")
save(ld_max_time,file="ld_max_time.RData")
save(tatum,file="tatum.RData")
save(pitch,file="pitch.RData")
save(timbre,file="timbre.RData")

save(features,file="features.RData")







###############################
######data cleaning###########
##############################


replace_na<-function(x){
  x[is.na(x)]<-mean(na.omit(x))
  x
}

features_cleaned<-apply(features,2,replace_na)
save(features_cleaned,file="features_cleaned.RData")


##############################################
######Devide train data and test data########
##############################################
smp_size<-floor(0.8*nrow(features_cleaned))
set.seed(123)
train_ind<-sample(seq_len(nrow(features_cleaned)),size=smp_size)
train_msc<-features_cleaned[train_ind,]
test_msc<-features_cleaned[-train_ind,]







#####################
#####topic models####
#####################
load("/Users/monicatao/Documents/ads/project4/Project4_data/lyr.RData")
lyr<-lyr[,-c(2,3,6:30)]

train_lyr<-lyr[train_ind,]
test_lyr<-lyr[-train_ind,]


#prepare documents for topic model

documents_t<-list()
for(i in 1:nrow(train_lyr)){
  c<-train_lyr[i,-1]
  documents_t[[i]]<-as.matrix(rbind(as.integer(which(c!=0)-1),as.integer(c[which(c!=0)])))
}
save(documents_t,file="documents_t.RData")



# MCMC and model tuning parameters:
library(lda)

K <- 20
G <- 1000
alpha <- 0.1
eta <- 0.1


set.seed(357)
t1 <- Sys.time()
fit_train <- lda.collapsed.gibbs.sampler(documents = documents_t, K = K, vocab = colnames(lyr[,-1]), 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2 - t1  # about 10 minutes on laptop

save(fit_train,file="lda_model.RData")


theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))


tm_top<-fit_train$document_sums
colnames(tm_top)<-train_ind
train_topic<-apply(tm_top,2,which.max)
t<-table(train_topic)




#############################################
######multinomial logistic regression########
#############################################

######PCA because in the multilogistic, it only allows less than 1000 weights
v<-apply(train_msc,2,var) #check the variance accross the variables
pca =prcomp(train_msc,scale = T, center = T) 

save(pcs,file="PCA.RData")

train_msc_pca<-train_msc %*% (pca$rotation[,1:50])


train_dat<-as.data.frame(cbind(train_msc_pca,train_topic))
train_dat$train_topic<-factor(train_topic)

#######
library(nnet)
form<-paste("train_topic ~",paste(colnames(train_dat[,1:50]),collapse = "+"))
mlogit <- multinom(form, data = train_dat)

save(mlogit,file="multilogit.RData")


test_msc_pca<-test_msc %*% (pca$rotation[,1:50])
pred<-predict(mlogit,newdata=test_msc_pca,"probs")

pred2<-cbind(pred,"7"=rep(0,nrow(pred)))
pred2<-pred2[,c(1:6,20,7:19)]

wordcount<-pred2 %*% (fit_train$topics)

wordrank<-apply(wordcount,1,function(x) rank(-x,ties.method = "random"))
wordrank<-t(wordrank)
save(wordrank,file="resultrank.RData")

lyrrank<-apply(lyr[,-1],1,function(x) rank(-x))
lyrrank<-t(lyrrank)


#####Calculating error rate
rate<-vector()
for(cc in seq(20,200,5)){
  a<-sort(wordcount[1,],decreasing = T)[1:cc]
  b<-sort(test_lyr[1,-1],decreasing = T)[1:cc]
  
  rate[1+(cc-20)/5]<-sum(is.na(match(names(a),names(b))))/cc
}





############################
######Random Forest##########
############################
install.packages("randomForest")
library(randomForest)

#####Random forest allows features to be many, so I chose 311 which explain 80 of variances
summary(pca)
train_msc_pca<-train_msc %*% (pca$rotation[,1:311])

train_dat_rf<-as.data.frame(cbind(train_msc_pca,train_topic))
train_dat_rf$train_topic<-factor(train_topic)
f <- as.formula(paste( "train_topic ~", paste(colnames(train_dat_rf[,1:311]), collapse = " + ")))

fit_randomforest_pca<-randomForest(f,data=train_dat_rf)

save(fit_randomforest_pca,file="fit_rf_pca.RData")
 
test_msc_pca<-test_msc %*% (pca$rotation[,1:311])
pred_rf<-predict(fit_randomforest_pca,newdata=test_msc_pca,"prob")

pred_rf2<-cbind(pred_rf,"7"=rep(0,nrow(pred_rf)))
pred_rf2<-pred_rf2[,c(1:6,20,7:19)]



wordcount_rf<-pred_rf2 %*% (fit_train$topics)

wordrank_rf<-apply(wordcount_rf,1,function(x) rank(-x,ties.method = "random"))
wordrank_rf<-t(wordrank_rf)

save(wordrank_rf,file="wordrank_rf.RData")


rate<-vector()
for(cc in seq(20,200,5)){
  a<-sort(wordcount_rf[1,],decreasing = T)[1:cc]
  b<-sort(test_lyr[1,-1],decreasing = T)[1:cc]
  
  rate[1+(cc-20)/5]<-sum(is.na(match(names(a),names(b))))/cc
}

rate



