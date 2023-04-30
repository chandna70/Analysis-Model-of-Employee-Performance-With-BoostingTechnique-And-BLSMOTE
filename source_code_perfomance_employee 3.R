library(caret)
library(dplyr)
library(xgboost)
library(Hmisc)
library(factoextra)
library(NbClust)
library(ggplot2)
library(ggcorrplot)
library(ggpubr)
library(gridExtra)
library(fastDummies)
library(smotefamily)
library(SHAPforxgboost)

##function

index_duplicate_data=function(data,columns)
  {
  df_duplicate=data[which(duplicated(data[,columns])==TRUE),]
  list_index=as.numeric(rownames(df_duplicate))
  return (list_index)
}

theme_black = function(base_size = 12, base_family = "") {
  
  theme_grey(base_size = base_size, base_family = base_family) %+replace%
    
    theme(
      # Specify axis options
      axis.line = element_blank(),  
      axis.text.x = element_text(size = base_size*0.8, color = "white", lineheight = 0.9),  
      axis.text.y = element_text(size = base_size*0.8, color = "white", lineheight = 0.9),  
      axis.ticks = element_line(color = "white", size  =  0.2),  
      axis.title.x = element_text(size = base_size, color = "white", margin = margin(0, 10, 0, 0)),  
      axis.title.y = element_text(size = base_size, color = "white", angle = 90, margin = margin(0, 10, 0, 0)),  
      axis.ticks.length = unit(0.3, "lines"),   
      # Specify legend options
      legend.background = element_rect(color = NA, fill = "black"),  
      legend.key = element_rect(color = "white",  fill = "black"),  
      legend.key.size = unit(1.2, "lines"),  
      legend.key.height = NULL,  
      legend.key.width = NULL,      
      legend.text = element_text(size = base_size*0.8, color = "white"),  
      legend.title = element_text(size = base_size*0.8, face = "bold", hjust = 0, color = "white"),  
      legend.position = "right",  
      legend.text.align = NULL,  
      legend.title.align = NULL,  
      legend.direction = "vertical",  
      legend.box = NULL, 
      # Specify panel options
      panel.background = element_rect(fill = "black", color  =  NA),  
      panel.border = element_rect(fill = NA, color = "white"),  
      panel.grid.major = element_line(color = "grey35"),  
      panel.grid.minor = element_line(color = "grey20"),  
      panel.margin = unit(0.5, "lines"),   
      # Specify facetting options
      strip.background = element_rect(fill = "grey30", color = "grey10"),  
      strip.text.x = element_text(size = base_size*0.8, color = "white"),  
      strip.text.y = element_text(size = base_size*0.8, color = "white",angle = -90),  
      # Specify plot options
      plot.background = element_rect(color = "black", fill = "black"),  
      plot.title = element_text(size = base_size*1.2, color = "white"),  
      plot.margin = unit(rep(1, 4), "lines")
      
    )
}

##Preparing_dataset
data_training=read.csv(file.choose(),header=T,na.string=c('','NA','na','\\N','\\n'))
data_testing=read.csv(file.choose(),header=T)
reference_data<-data_testing$Best.Performance
data_testing<-data_testing[,!(colnames(data_testing)%in%c('X','Best.Performance'))]
y_train<-data_training$Best.Performance
data_training<-data_training[,!(colnames(data_training)%in%c('X','Best.Performance'))]
data_training$Best.Performance<-y_train



len_train=nrow(data_training)
len_test=nrow(data_testing)
cat('Number of sample size for training data :',len_train,'\nNumber of sample for testing data:',len_test)

summary(data_training)

head(data_training)
##Check_NA_from_data_training
glimpse(data_training)
glimpse(data_testing)

df_na_train<-data_training%>%
  select_all()%>%
  summarise_all(funs(sum(is.na(.))))

df_na_test<-data_testing%>%
  select_all()%>%
  summarise_all(funs(sum(is.na(.))))

t(df_na_train)
t(df_na_test)


x<-df_train%>%
  select_all()%>%
  summarise_all(funs(sum(is.na(.))))
t(x)
##Target Class

prop.table(table(data_training$Best.Performance))

##Get categorical data

data_training$gender<-as.character(unlist(data_training$gender))
data_testing$gender<-as.character(unlist(data_testing$gender))

data_training$gender[data_training$gender=="1"]="M"
data_training$gender[data_training$gender=="2"]="F"

data_testing$gender[data_testing$gender=="1"]="M"
data_testing$gender[data_testing$gender=="2"]="F"

cat_train<-data_training%>%select_if(.,is.character)
cat_test<-data_testing%>%select_if(.,is.character)
describe(cat_train)
describe(cat_test)

##Get Numeric Data
num_train<-data_training%>%select_if(.,is.numeric)
num_test<-data_testing%>%select_if(.,is.numeric)
describe(num_train)
describe(num_test)

##Visualize categorize class to target class
cat_train['target']=data_training["Best.Performance"]
p=paste("p",paste(1:6),sep="")
cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
for (i in 1:(length(cat_train)-1)){
  a<-dplyr::select(cat_train[c(i,7)])%>%
    count(cat_train[c(i,7)],name='val')
  a["target"]=as.character(unlist(a["target"]))
  assign(p[i],ggplot(a,aes_string(x="target",y="val",fill=colnames(a[1])))+
    geom_bar(stat = "identity", position = "dodge")+
    scale_x_discrete("Target Class")+scale_y_continuous("Number of target")+scale_fill_manual(values=cbbPalette)+
    theme_minimal())
}
ggarrange(p1,p2,p3,p4,p5,p6,nrow=2,ncol=3)

for ( i in 1:(length(cat_train)-1)){
  a<-dplyr::select(cat_train[c(i)])%>%
    count(cat_train[c(i)],name='val')%>%mutate(dataset="training")
  b<-dplyr::select(cat_test[c(i)])%>%
    count(cat_test[c(i)],name='val')%>%mutate(dataset="testing")
  a<-rbind(a,b)
  assign(p[i],ggplot(a,aes_string(x="dataset",y=colnames(a[1])))+
    geom_point(aes_string(size="val",colour=colnames(a[1])))+
      scale_x_discrete("Dataset")+scale_colour_manual(values=cbbPalette)+theme_minimal())
}
ggarrange(p1,p2,p3,p4,p5,p6,nrow=2,ncol=3)


##plot the continous data

#distribution
p=paste("p",paste(1:length(num_train[,!(colnames(num_train)=="Best.Performance")])),sep="")
for (i in 1:length(p)){
assign(p[i],ggplot(num_train,aes_string(x=colnames(num_train[i])))+
  geom_histogram(alpha=2,aes(y=..density..),bins=30,fill="blue",col='black')+
  geom_density(alpha=0.3,fill='yellow',
               aes_string(x=colnames(num_train[i])),
               col='white',size=0.5,inherit.aes = FALSE)+
  scale_y_continuous("Dense")+theme_black())}
ggarrange(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,nrow=3,ncol=5)
#boxplot
for (i in 1:length(p)){
assign(p[i],ggplot(num_train,aes_string(x=colnames(num_train[i])))+
    geom_boxplot(col="white",fill="blue",outlier.colour = "#F0E442",lwd=.5)+
    stat_boxplot(geom = 'errorbar',col="white",lwd=.5)+theme_black())
}
ggarrange(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,nrow=3,ncol=5)

#continous->target
for(i in 1:length(p)){
a<-cbind(num_train[i],num_train["Best.Performance"])
a["Best.Performance"]=as.character(unlist(a["Best.Performance"]))
assign(p[i],ggplot(a,aes_string(x=colnames(a[1]),fill="Best.Performance"))+
geom_density(alpha=0.3,col="#999999")+
  scale_fill_manual(name="Class",values=c("blue","yellow"))+
  theme_black())}
ggarrange(p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,nrow=3,ncol=5)


##Engineering part

df=rbind(data_training[,!(colnames(data_training)=="Best.Performance")],data_testing)
##imputation NA value
df$Last_achievement_.[is.na(df$Last_achievement_.)]=mean(df$Last_achievement_.,na.rm=T)
df$Achievement_above_100._during3quartal[is.na(df$Achievement_above_100._during3quartal)]=mean(df$Achievement_above_100._during3quartal,na.rm=T)

list_col=colnames(cat_train[,!(colnames(cat_train)=="target")])
#make dummy
df_temp=df
for (i in list_col){
  df_temp<-dummy_cols(df_temp,select_columns = i,remove_selected_columns = TRUE)
}
df_train<-cbind(df_temp[1:len_train,],Best.Performance=data_training$Best.Performance)
df_test<-df_temp[(len_train+1):(len_test+len_train),]

##Turn to data frame
df_train=data.frame(df_train)
df_test=data.frame(df_test)

##corplot
mat_cor=cor(df_train[,(!colnames(df_train)=="Best.Performance")])
ggcorrplot(mat_cor,type = "lower",tl.cex = 7.5 )+rotate_x_text(angle= 90)

##Visualization K's value
fviz_nbclust(df_train[,!(colnames(df_train)=="Best.Performance")],kmeans,method='silhouette')+theme_black()

##Reconstruct_from_imbalance_data
prop.table(table(df_train$Best.Performance))
data_training_new<-BLSMOTE(df_train[,(!colnames(df_train)=="Best.Performance")],df_train$Best.Performance,K=2)
data_training_new<-data_training_new$data
prop.table(table(data_training_new$class))
nrow(data_training_new)


##PCA
#standarize data
std_x_train = scale(data_training_new[,(!colnames(data_training_new)=="class")],
                    scale=T,
                    center = T)
pca_train = prcomp(std_x_train,center=F,scale=F)
summary(pca_train)
df_pca_train<-data.frame(pca_train$x)

std_x_test=scale(df_test,center=attr(std_x_train,"scaled:center"),
                 scale = attr(std_x_train,"scaled:scale"))
df_pca_test=data.frame(std_x_test %*% pca_train$rotation)

##make x_train,y_train,x_tesst,y_test
x_train<-data_training_new[,(!colnames(data_training_new)=="class")]
y_train<-as.numeric(data_training_new$class)
x_test<-df_test
y_test<-reference_data

x_train<-df_pca_train
y_train<-as.numeric(data_training_new$class)
x_test<-df_pca_test
y_test<-reference_data
##Modelling

#xgboost
x_train_xgb<-xgb.DMatrix(as.matrix(x_train),label=y_train)
x_test_xgb<-xgb.DMatrix(as.matrix(x_test),label=y_test)

params_xgb<-list(booster = "dart", 
                 objective = "binary:logistic",eta=0.3,gamma=1,max_depth=5,
                 min_child_weight=2,subsample=1,colsample_bytree=1,lambda=1.25,alpha=0.75)
xgb_cv<-xgb.cv(params=params_xgb,
               data=x_train_xgb,nrounds=600,nfold=5,showsd = T, 
               early.stop.round = 35, maximize = F,metrics=c('auc'))
gb_dt <- xgb.train(params = params_xgb,
                   data = x_train_xgb,
                   nrounds = xgb_cv$best_iteration,
                   print_every_n = 2,
                   eval_metric=c('auc'),
                   watchlist=list(train=x_train_xgb,eval=x_test_xgb))

xgb_pred<-predict(gb_dt,x_test_xgb)
binary_pred_xgb<-as.numeric(xgb_pred > 0.5)
confusionMatrix(as.factor(binary_pred_xgb),as.factor(y_test))

##Xgboost featuring importance
xgb_contrib<-shap.values(gb_dt,as.matrix(x_train))
shap_value_xgb<-xgb_contrib$shap_score
shap_long_xgb<-shap.prep(shap_contrib = shap_value_xgb,
                         X_train = as.matrix(x_test),
                         var_cat=as.matrix(as.facreference_data))
shap.plot.summary(shap_long_xgb,scientific = F)
shap.plot.force_plot()

important_var<-shap.importance(shap_long_xgb,top_n=10)
important_var<-important_var%>%dplyr::select(variable,mean_abs_shap)%>%
  mutate(percentage=round(mean_abs_shap*100,2))
ggplot(important_var)+geom_bar(aes(x=variable,y=percentage),stat='identity')+
  scale_x_discrete(guide=guide_axis(angle=90))+
  geom_text(aes(label=paste(percentage,"%"),x=variable,y=percentage+1.5))+
  ggtitle('Top 10 important variable')

shap.prep
shap.importance(shap_long_xgb,top_n=10)
xgb_pred
