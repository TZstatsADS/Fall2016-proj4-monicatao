library(rhdf5)
load("~/rf_fit.RData")
load("~/PCA_fit.RData")
load("~/tm_fit.RData")
load("~/Documents/ads/project4/Project4_data/lyr.RData")


#######Estracting Features#############
dir<-"/Users/monicatao/Documents/ads/project4/Project4_data/TestSongFile100/"
files <- paste0("/Users/monicatao/Documents/ads/project4/Project4_data/TestSongFile100/testsong",1:100)
files <- paste0(files,".h5")
filename <- gsub(".*/", "", files)
test_filename <-list()
for (i in 1:100) {
  test_filename[[i]] <- strsplit(filename[i],".",fixed = TRUE)[[1]][1]
}
filename <- unlist(test_filename)

bar<-list()
beat<-list()
ld_max<-list()
ld_max_time<-list()
pitch<-list()
timbre<-list()
tatum<-list()


for(y in files){
  sound<-h5read(y, "/analysis")
  bar[[y]]<-sound$bars_start
  beat[[y]]<-sound$beats_start
  ld_max[[y]]<-sound$segments_loudness_max
  ld_max_time[[y]]<-sound$segments_loudness_max_time
  pitch[[y]]<-sound$segments_pitches
  timbre[[y]]<-sound$segments_timbre
  tatum[[y]]<-sound$tatums_start
}

H5close()

feat<-list(bar,beat,ld_max,ld_max_time,tatum)
features<-matrix(nrow=0,ncol=length(feat)*7+12*21*2)
for(i in 1:100){
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
rownames(features)<-filename
sum(is.na(features)) ####0

save(features,file="test_features.RData")

#################################
########Test Data###############
################################
features_pca<-features %*% (pca_fit$rotation[,1:230])
pred_rf<-predict(rf_fit,newdata=features_pca,"prob")


wordcount_rf<-pred_rf %*% (tm_fit$topics)

wordrank_rf<-apply(wordcount_rf,1,function(x) rank(-x,ties.method = "min"))
wordrank_rf<-t(wordrank_rf)
wordrank_rf<-cbind(wordrank_rf,matrix(4987,nrow=100,ncol=27))
colnames(wordrank_rf)[4974:5000]<-colnames(lyr)[c(2,3,6:30)]

wordrank_rf<-wordrank_rf[,c(4974:4975,1:2,4976:5000,3:4973)]

write.csv(wordrank_rf,"result.csv")






