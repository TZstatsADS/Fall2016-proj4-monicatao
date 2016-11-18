library(rhdf5)
load("/Users/monicatao/Documents/ads/project4/ads_proj4/lda_model.RData")
load("/Users/monicatao/Documents/ads/project4/ads_proj4/PCA.RData")
load("/Users/monicatao/Documents/ads/project4/ads_proj4/fit_rf_pca.RData")


#######Estracting Features#############
dir<-"/Users/monicatao/Documents/ads/project4/Project4_data/TestSongFile100/"
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
rownames(features)<-files
sum(is.na(features)) ####0



#################################
########Test Data###############
################################
features_pca<-features %*% (pca$rotation[,1:311])
pred_rf<-predict(fit_randomforest_pca,newdata=features_pca,"prob")

pred_rf2<-cbind(pred_rf,"7"=rep(0,nrow(pred_rf)))
pred_rf2<-pred_rf2[,c(1:6,20,7:19)]


wordcount_rf<-pred_rf2 %*% (fit_train$topics)

wordrank_rf<-apply(wordcount_rf,1,function(x) rank(-x,ties.method = "random"))
wordrank_rf<-t(wordrank_rf)

write.csv(wordrank_rf,"result.csv")






