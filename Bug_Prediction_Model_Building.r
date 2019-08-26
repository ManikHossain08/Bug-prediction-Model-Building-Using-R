library(foreign)
library(caret)
library(car)
library(nlme)
library(rms)
library(e1071)
library(BiodiversityR)
library(moments)
library(randomForest)
library(ROCR)
library(pROC)
library(DMwR)
library(vioplot)
library(Hmisc)
library(psych)
library(effsize)

#Here are some sample questions:

#1. Do files with post-release bugs have higher complexity than files without post-release bugs? 
# EXAMPLE: files with more than 100 LOC and less than 100 LOC which files has more post_bugs?
#2. Size and pre-release defects, which has a higher correlation with post-release bugs?
#3. How well can you model post release bugs? What are the top 5 metrics in model fit? 
                                          #What are the topic 5 metrics in model output?
#4. How well can you predict post release bugs? 

data<-read.csv("./qt50.csv")

summary(data)

# see which data is not numerical and think if you want to keep the data or not.
# remove component and subsystem
drop=c("comp","subsystem","mean_discussion","mean_revspeed")
data=data[,!(names(data) %in% drop)]

names(data)
summary(data)

# Sample exam question 1: Do files with post-release bugs have higher complexity than files without post-release bugs?
# need data with and without postrelease bugs: post_bugs, so make subsets 
withbug=subset(data, data$post_bugs>0)
withoutbug=subset(data, data$post_bugs==0)

# now to compare we need to use a t-test or wilcox test but first you need to find out if data is normally distrib or not
summary(data$complexity)
#since the median and mean are close, they are close to normal distrib so you could use t-test but see the histogram
plot(hist(data$complexity))
#boxplot(data$complexity)
#after seeing the outliers in the histogram you need to use Wilcox test

t.test(withbug$complexity, withoutbug$complexity)

wilcox.test(withbug$complexity, withoutbug$complexity)
# we know now they are different so we need to use the size

cliff.delta(withbug$complexity, withoutbug$complexity)
# means the difference between the 2 is medium -see the first line from Cliff's Delta
# so because the p valu is < 0.05 then the answer to the first question is yes
# the cliff delta tells how big is the size difference
# if the p value is > 0.05 we don't know anything as it is uncertain so it does not make sense to measure cliff.d
# so if p value were higher then the answer98ikjm  would have been: There is no way we can tell (because the noise is too big)

summary(data$size)
# this means we need to do a wilcox test for sure since mean is very different from median
wilcox.test(withbug$size, withoutbug$size)
#wilcox.test(withoutbug$size,withbug$size) #Just Curiosity
# p value is super small, so we are certain they are different
cliff.delta(withbug$size, withoutbug$size)
#cliff.delta(withoutbug$size,withbug$size)   #Just Curiosity
# the first one is lower and the difference is large

skewness(data$size)
kurtosis(data$size)
# it tells it is skewed to left, so not normally distributed

# Sampole question: Which files from files with more than 100 LOC and less than 100 LOC has more post_bugs?

bigfile=subset(data, data$size>100)
smallfile=subset(data, data$size<=100)

summary(bigfile)
summary(smallfile)

# now we need to to compare bugs
summary(data$post_bugs)
plot(hist(data$post_bugs))
skewness(data$post_bugs)
t.test(bigfile$post_bugs, smallfile$post_bugs) #p-value is very high
#left(MMH: not sure,may be right  bcz its plus(+)) skewed so we use wilcox
wilcox.test(bigfile$post_bugs, smallfile$post_bugs)
#wilcox is small so there is different
cliff.delta(bigfile$post_bugs, smallfile$post_bugs)
# so the difference is very small

#Sample question2: Size and pre-release(Prior_bug) defects, which has a higher correlation with post-release bugs?
cor(data$size, data$post_bugs, method = "spearman") # pearson is value based dont care about rank
cor(data$prior_bugs, data$post_bugs, method = "spearman") #bcz we have lot of outlier using histogram of post bug.
#so the answer is prior_bugs has the higher correlation
# they are very close, NOTE: by default is Pearson correlation, but since the data is not normally distributed use Spearman

summary(data$size)
summary(data$prior_bugs)
skewness(data$size)
skewness(data$prior_bugs)

# Answer OF QUESTION2 :  Prior bugs has a higher correlation AS THE VALUE IS LITTLE BIT HIGHER, BUT although there are almost the same VALUE.

#Question 3: How well can you model post release bugs? What are the top 5 metrics in model fit? What are the topic 5 metrics in model output?

#building the model for post_bugs, so I need to take it out from the data bcz it would be dependent variable
drop2=c("post_bugs")
independant = data[,!(names(data) %in% drop2)]
# check if any of the independant features are not highly correlated with the target or if not the same, 
#then you manually remove them before doing the actual correlation calculation on the feature set
# in our sample data we did not have this situation so it's not present here.

#remove higher correlation - basically columns which say the same story
correlations<-cor(independant, method = "spearman")
#find the higher correlation and set cutoff between 0.7 and 0.8
highCorr <- findCorrelation(correlations, cutoff = 0.75) #correlation analysis to remove
# this link tells you how its remove high correlation : https://www.listendata.com/2015/06/simplest-dimensionality-reduction-with-r.html
#so the highCorr has all the columns number which need to be removed
# now keep only the column names you need which has lower correlation 
low_cor_names=names(independant[,-highCorr])
#get the data
low_cor_data= independant[(names(independant) %in% low_cor_names)]
#double check your new datase
cor(low_cor_data)

# next step is redundancy analysis - combination of columns which say the same story, but one to many relation 
dataforredun=low_cor_data 
redun_obj = redun (~. ,data = dataforredun ,nk = 0) # redundancy analysis to remove the higer p-value
# the nk=0 means a linear model. It means how many turning points Since we build a linear model it makes no sense to have nk>0

#look at the data
redun_obj
names(redun_obj)
#remove redundancies
after_redun= dataforredun[,!(names(dataforredun) %in% redun_obj$Out)] # again out the redundant data, like find highcorrelation analysis
#the after_redun is the data ready for modeling and we now need a formula
names(after_redun) #now you have original data to work with
# concatenate all the strings wiht post_bugs>0
form=as.formula(paste("post_bugs>0~",paste(names(after_redun),collapse="+"))) # we extract the after_redun data frame for build the formula
# or type it....post_bugs>0~size+complexity+change_churn+....

# model is logistic regrestion and WE NEED to LOG the DATA
# WATCH that you need to use the data which has the post_bugs
model=glm(formula=form, data=log10(data+1), family = binomial(link = "logit")) 
# this formula(form) will work on real data to find linear relation that return true or false
#it is basically logistic regression that always provide output in this way change or no change (like yes or no, true or false)
# use log(data+1) cause you may have data which is 0 and that's going to throw an error

summary(model) # from here star(*) marked metrics will build new formula for linear regression again.
anova(model) #
# tells you for every metric how significant it is -> the stars(*,**,***) next to the features
#form2=post_bugs>0~size+voter_ownership
newform=post_bugs>0~size+voter_ownership+little_discussion+median_discussion+ minor_wrote_no_major_revd+ major_actors+ median_minexp
newmodel=glm(formula=newform, data=log10(data+1), family = binomial(link = "logit"))

summary(newmodel)
#so now we see the model only has features with stars so we're good(becz every metrics with stars(*))
# so Questions how well you can model post release bug that means asking for model fit = 1-(residual/null)
# calculation = 1-(733.18/1300.92) =0.436, this is basically R2(square)
# here the value is above 0.4 so this is very good logistic model.
# what is null deviance and residual deviance?

#ow well can you model post release bugs
# 1-residual/null from the Model!!! (model fit value)
#For logistic model anything above 0.2 is good. For linear/logistic we need above 0.4 for efectiveness

#What are the top 5 metrics in model fit? Model fit is the deviance, the R-square(answer)
anova(newmodel)
#newmodel <- newmodel[order(newmodel$deviance),] #order
# gives you a table that tells you which are the top contributors to model fit



#What are the topic 5 metrics in model output?
# the output is having a bug or not. So what is the probability of having a bug?
# we have 7 metrics so we do a prediction with the metrics mean value. On the mean value what is the probability of having a bug
# so we can answer qustions like, if my file size is double than mean size, what is the chance of having a bug?
# the last questions is about prediction, see above line says that things, he can ask you this Q in exam. 
#collect the means of the features that are significant into a new data frame and make a prediction for model o/p
testdata=data.frame(size= log10(mean(data$size)+1), voter_ownership =log10(mean(data$voter_ownership)+1), little_discussion=log10(mean(data$little_discussion)+1), median_discussion=log10(mean(data$median_discussion)+1), minor_wrote_no_major_revd=log10(mean(data$minor_wrote_no_major_revd)+1), major_actors=log10(mean(data$major_actors)+1), median_minexp=log10(mean(data$median_minexp)+1))
predict(newmodel,testdata, type="response")#BASELINE!!!! 
#that is consider as a standard value then we have to compare rest of the
#-changed predict value to find out the absoulute value for topic model output
# the chance of having a bug is 0.39 for a file having the means as input
# this is the typical case, the baseline, BASELINE!!!!

# now we alter the typical case and double the size
testdata2=data.frame(size= log10(mean(data$size)*2+1), voter_ownership =log10(mean(data$voter_ownership)+1), little_discussion=log10(mean(data$little_discussion)+1), median_discussion=log10(mean(data$median_discussion)+1), minor_wrote_no_major_revd=log10(mean(data$minor_wrote_no_major_revd)+1), major_actors=log10(mean(data$major_actors)+1), median_minexp=log10(mean(data$median_minexp)+1))
predict(newmodel,testdata2, type="response")
# so by doubling the file size the probability increases to 0.4498

testdata3=data.frame(size= log10(mean(data$size)+1), voter_ownership =log10(mean(data$voter_ownership)*2+1), little_discussion=log10(mean(data$little_discussion)+1), median_discussion=log10(mean(data$median_discussion)+1), minor_wrote_no_major_revd=log10(mean(data$minor_wrote_no_major_revd)+1), major_actors=log10(mean(data$major_actors)+1), median_minexp=log10(mean(data$median_minexp)+1))
predict(newmodel,testdata3, type="response")
#so by increasing the ownership, the probability decreases by half

testdata4=data.frame(size= log10(mean(data$size)+1), voter_ownership =log10(mean(data$voter_ownership)+1), little_discussion=log10(mean(data$little_discussion)*2+1), median_discussion=log10(mean(data$median_discussion)+1), minor_wrote_no_major_revd=log10(mean(data$minor_wrote_no_major_revd)+1), major_actors=log10(mean(data$major_actors)+1), median_minexp=log10(mean(data$median_minexp)+1))
predict(newmodel,testdata4, type="response")
#so by increasing the ownership, the probability decreases by half

# so we need to manually calculate for every metric and see which are the highest changes compared to the BASELINE!!!!
###################################################
# Sample question 4: How well can you predict post release bugs? 
# we need to calculate precision and recall
# so we need to split data in test and train

# let's pick 2/3 in training and 1/3 in test
training_size=round(2*length(data$size)/3,digits=0)
testing_size=length(data$size)-training_size
training_index=sample(nrow(data), training_size)

training=data[training_index,]
testset=data[-training_index,]

# now you build the model with the training data
drop2=c("post_bugs")
independant = data[,!(names(training) %in% drop2)]
#remove higher correlation - basically columns which say the same story
correlations<-cor(independant, method = "spearman")
#find the higher correlation and set cutoff between 0.7 and 0.8
highCorr <- findCorrelation(correlations, cutoff = 0.75)
#so the highCorr has all the columns which need to be removed
# now keep only the column names you need
low_cor_names=names(independant[,-highCorr])
#get the data
low_cor_data= independant[(names(independant) %in% low_cor_names)]
#double check your new datase
cor(low_cor_data)

# next step is redundancy analysis - combination of columns which say the same story
dataforredun=low_cor_data
redun_obj = redun (~. ,data = dataforredun ,nk = 0)
# the nk=0 means a linear model. It means how many turning points Sincew we build a linear model it makes no sense to have nk>0

#look at the data
redun_obj
#remove redundancies
after_redun= dataforredun[,!(names(dataforredun) %in% redun_obj $Out)]
#the after_redun is the data ready for modeling and we now need a formula
names(after_redun)
# concatenate all the strings wiht post_bugs>0
form=as.formula(paste("post_bugs>0~",paste(names(after_redun),collapse="+")))
# or type it....post_bugs>0~size+complexity+change_churn+....

# model is logistic regrestion and WE NEED to LOG the DATA
# WATCH that you need to use the data which has the post_bugs
#model=glm(formula=form, data=log10(data+1), family = binomial(link = "logit")) #need correction, data would be trainingset
model=glm(formula=form, data=log10(training+1), family = binomial(link = "logit"))
# now evaluate on test
predictions<-predict(model, log10(testset+1), type="response")
#precision is predicted as buggy and is actually buggy
#calculate true positives
TP=sum((predictions>0.5) & (testset$post_bugs>0))
precision=TP/sum((predictions>0.5))
precision

recall=TP/ sum(testset$post_bugs>0)
recall

library(pROC)
roc_obj<-roc(testset$post_bugs>0, predictions)
auc(roc_obj)
# AUC is very high so we got a very good model!

