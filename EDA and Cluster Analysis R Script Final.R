###Exploring the Patient Population at Bruner Clinic
##This is the R code for the EDA process of exploring the patient population at Bruner Clinic in Denver, CO 
#with a pleothra of different chronic conditions. This first set of code is just what I did to learn about the data
#I have and what I can take away from a basic EDA process with this dataset and which variables do I think would be
#interesting for further analysis.

##Load & View the data
#Can't analyze data if we don't have any, I first need to start by loading the dataset I will be analyzing.
library(readr)
bpp <- read_csv("Data Science School Documents/MSDS 69 Da6ta Science Practicum II/BrunerPatientPopulation.csv")
head(bpp)

##Load the libaries
#There are a lot of libraries in RStudio, here are some of the ones I'll be using for this EDA process
install.packages("dplyr")
library(dplyr)
install.packages("ggplot2")
library(ggplot2)

##Basic EDA
#Start by just learning about the characteristics of the dataset. I was the one who built the dataset and brought in 
# what fields I believed to be useful from different sources, but what does RStudio say about what I'm bringing in.
dim(bpp)
#1117 records and 78 fields in total
colnames(bpp)

#What is the data source's structure?
str(bpp)
#There appears to be some variables with the wrong data type, and some variables with the null values.
#Let's start by changing the data types

##Change the data types of specific variables within the Bruner Data:
#Start with variables that need to be changed from intergers to character types
bpp$ENC_TYPE_C <- as.character(bpp$ENC_TYPE_C)
bpp$DEPARTMENT_ID <- as.character(bpp$DEPARTMENT_ID)
bpp$LINE <- as.character(bpp$LINE)
bpp$SERV_AREA_ID <- as.character(bpp$SERV_AREA_ID)
bpp$CUR_PRIM_LOC_ID <- as.character(bpp$CUR_PRIM_LOC_ID)
bpp$DX_ID <- as.character(bpp$DX_ID)

#Now the variables that are characters and need to change to numeric variables, there are fortunately only family income and size for these that fields that need changed.
bpp$FAMILY_INCOME <- as.numeric(bpp$FAMILY_INCOME)
bpp$FAMILY_SIZE <- as.integer(bpp$FAMILY_SIZE)

#Setting variables to factors
bpp$ICD_Group <- as.factor(bpp$ICD_Group)
bpp$LOC_NAME <- as.factor(bpp$LOC_NAME)
bpp$ZIP <- as.factor(bpp$ZIP)
bpp$SEX_C <- as.factor(bpp$SEX_C)
bpp$ETHNIC_GROUP_C <- as.factor(bpp$ETHNIC_GROUP_C)
bpp$MARITAL_STATUS_C <- as.factor(bpp$MARITAL_STATUS_C)
bpp$LANGUAGE_C <- as.factor(bpp$LANGUAGE_C)
bpp$TOBACCO_USE_VRFY_YN <- as.factor(bpp$TOBACCO_USE_VRFY_YN)

#I want to start by looking at my data and making sure I made all the requisite changes to the data types.
str(bpp)
#My data types look good!

#Now I want to make sure my data was uploaded into Rstudio without any concerns so lets check that using the head & tail funcitons
head(bpp)
tail(bpp)
#I have noticed some columns with what appears to be a lot of null values, let's take a deeper look
# and try to figure out if there are only null values or if there is some useable values in these variables.

##Explore the variables that are seemingly filled with only nulls:
#DEPARTMENT_ID
sum(is.na(bpp$DEPARTMENT_ID))
plot(bpp$DEPARTMENT_ID)
table(bpp$DEPARTMENT_ID)
#Though we have null values, it appears there are indeed some records with actual department ID's, at SCL these are numbered so plotting them is plausable when showing 
# if you have any of these values or not.

#CHECKIN_TIME
sum(is.na(bpp$CHECKIN_TIME))
table(bpp$CHECKIN_TIME)
#Creating a table for check-in times showed that there are indeed times recorded for when patient's were checked into the hospital but just not all records have this recorded.

#CHECKOUT_TIME
sum(is.na(bpp$CHECKOUT_TIME))
table(bpp$CHECKOUT_TIME)
#The same strategy for check-in time was used here for check-out time, but when comparing the tables, it's evident that there isn't many times where check-out time was recorded.

#ENC_CLOSE_DATE
sum(is.na(bpp$ENC_CLOSE_DATE))
table(bpp$ENC_CLOSE_DATE)
#Again, same strategy can be used since ENC_CLOSE_DATE is a date-time, there isn't many but there appears to be about 1 time where this value was recorded.

#PAT_PAIN_SCORE_C
sum(is.na(bpp$PAT_PAIN_SCORE_C))
bpp$PAT_PAIN_SCORE_C <- as.factor(bpp$PAT_PAIN_SCORE_C)
table(bpp$PAT_PAIN_SCORE_C)
plot(bpp$PAT_PAIN_SCORE_C)
#Unfortunately, the PAT_PAIN_SCORE_C had all null values

#FAMILY_SIZE
sum(is.na(bpp$FAMILY_SIZE))
table(bpp$FAMILY_SIZE)
plot(bpp$FAMILY_SIZE)
#Unfortunately family size only yields null results for the dataset that I have to analsis. I will be omitting this from my dataset later.

#FAMILY_INCOME
sum(is.na(bpp$FAMILY_INCOME))
table(bpp$FAMILY_INCOME)
plot(bpp$FAMILY_INCOME)
#Same thing with family income, no non-null variables so this variable will be omitted.

#PAT_HOMELESS_YN
sum(is.na(bpp$PAT_HOMELESS_YN))
table(bpp$PAT_HOMELESS_YN)
#Isn't null, but with only three values in the variable, all of which are null it makes for a kind of pointless variable for this set of patients.

#DEF_FIN_CLASS_C
sum(is.na(bpp$DEF_FIN_CLASS_C))
table(bpp$DEF_FIN_CLASS_C)
plot(bpp$DEF_FIN_CLASS_C)
#Only null values, so we can get rid of this variable as well.

#After looking into some of my variables that I suspected to be strictly null values I came back with some encouraging and some not so much, it looks like there are a number
# of variables that it would be best for me to get out of my dataset before moving on with my analysis, I luckily can do that here with RStudio.

##Omitting Null variables from my dataset
#The columns that I am going to omit from my dataset are FAMILY_SIZE, FAMILY_INCOME, PAT_HOMELESS_YN, DEF_FIN_CLASS_C, PAT_PAIN_SCORE_C & ENC_CLOSE_DATE
#To do this I need to make sure of the location for each of these variables within the dataset, so I don't omit anything I want to keep in there.
#I can find the location just by looking at the names of all my variables.
names(bpp)
#From the looks of my data columns here is the locations ENC_CLOSE_DATE = 14; PAT_PAIN_SCORE_C = 26; FAMILY_SIZE = 27; FAMILY_INCOME = 28; PAT_HOMELESS_YN = 29; DEF_FIN_CLASS_C = 45
#Let's make a subset of this data before omitting it
osubset <- select(bpp, DEPARTMENT_ID, CHECKIN_TIME, CHECKOUT_TIME, ENC_CLOSE_DATE, PAT_PAIN_SCORE_C, FAMILY_SIZE, FAMILY_INCOME, PAT_HOMELESS_YN, DEF_FIN_CLASS_C)
head(osubset)
str(osubset)
#It truly appears none of the data from the osubset will be helpful for my analysis going further so it is time to get rid of it.
bpp <- select(bpp, -(DEPARTMENT_ID), -(CHECKIN_TIME), -(CHECKOUT_TIME),-(ENC_CLOSE_DATE), -(PAT_PAIN_SCORE_C), -(FAMILY_SIZE), -(FAMILY_INCOME), -(PAT_HOMELESS_YN), -(DEF_FIN_CLASS_C))

str(bpp)
head(bpp)
#Now our patient population has 69 variables.

#Then with our remaining variables that are null were just goin to omit the null values.
#I would like to make a subset of some of the data I have left now, singling out perhaps which variables I want to measure and which I want to aggregate upon.
#I will start with a subset of my variables that are factors, I want to make sure I know which ones these are so I can then compare my different factors across different measures and eventually use 
# these to help with my visualizations and cluster analysis.
#I want to start again by looking at my column names, I always find this to be an effect way to point out different variables I want to subset.
names(bpp)
#Some of these variables are pointless for my subset, even though they aren't measures. The reason is because they were good for connecting my data, and great for primary keys for possible counting 
# purposes, but are unnecessary for what I'm doing here, it could be another situation where I can omit them from my entire analysis, but first let me make my two subsets.
#The variables I'd like to look into are: CONTACT_DATE, LOC_NAME, CURRENT_ICD10_LIST, ICD Group, TOBACCO_USE_VRFY_YN, CITY, State, Zip Codes Clean, County, BIRTH_DATE,
# SEX_C, ETHNIC_GROUP_C, MARITAL_STATUS_C, LANGUAGE_C

#Now I'm curious to see what some individual records look like for our bpp dataset, we can do this pretty simply.
bpp[c(1,3,5,7,9)]
#The command above is great for looking at data rows, but unfortunately I can't do that effectively because of the amount of variables 
# so what might be the best course of action is to split them into 2 subsets, one for dimensional variables, and one for measures

#The subset of these variables:
dim_subset <- select(bpp, ICD_Group, CONTACT_DATE, LOC_NAME, CURRENT_ICD10_LIST, TOBACCO_USE_VRFY_YN, CITY, State, ZIP, County, BIRTH_DATE, SEX_C, ETHNIC_GROUP_C, MARITAL_STATUS_C, LANGUAGE_C)
head(dim_subset)
str(dim_subset)
#Looking into the structure of this subset, it looks like we have a mix of variables that are character based and structure based, The ones that are factor based range from just one level with the Tobacco field
# to 750 with the zip code field. Ultimately, the variables which are factors will be what I'm most interested in when it comes to variables within this subset, the reason why is it appears they will be what will 
# help me better understand the patient population at Bruner Clinic and the chronic conditions which are the ICD_Group is one that we really want to highlight to understand if some of these variables are correlated to 
# why patients are getting these conditions.

#Now let's take a look at some of these dimensions 
colnames(dim_subset)
#There appears to be only 14 variables within this subset, only a few that I think are important going forward, let's take a quick of those individually.
dim_subset[,1]
table(dim_subset$ICD_Group)
#Looking ath the first few samples of the dim_subset you'd assume that there is a lot for hypertension, but when tabled, it is apparent that Arthritis is 
# far and away the highest numbered response for the chronic condition these patients come in with per our population. 
dim_subset[,8]
table(dim_subset$ZIP)
#Our zip codes are very wide ranging, which means our patients are coming from more than just places within Colorado, a little surprising 
# with how vast but the majority are within Colorado area codes starting with 80 and 81 so that makes sense for the data collected.
dim_subset[,11]
#The 11th column is primarily 1 and 2 which is 1 for female and 2 for male. 
table(dim_subset$SEX_C)
#The only values I'm actually seeing here are 1's and 2's nothing else for the gender of the patients, with almost 2000 more females than males

#Now let's make a subset of the measures, 
meas_subset <- select(bpp, BP_SYSTOLIC:Uninsured_Tract)
head(meas_subset)
#Looks like I have some measures I still need to disgard.
meas_subset <- select(meas_subset, -(HEIGHT), -(PHYS_BP), -(TOBACCO_USE_VRFY_YN))
meas_subset <- select(meas_subset, -(CITY:LANGUAGE_C))
colnames(meas_subset)
meas_subset <- select(meas_subset, -(ICD_Code:CUR_PRIM_LOC_ID))
str(meas_subset)
#There is a total of 28 variables of interest within this subset from the Bruner Patient data. It is mix of numerical and binary data which gives insight on the patient's health based on their encounter and a number of 
# variables which are tied to the zip codes and counties these patients are from to help pull in insights into the environments patients are from and what that could be contributing to their overall health.

##Some of the measure columns are different in how they are counted within our measures subset, I would like to take a look at some of these and 
# assess how they're different and how I can use them later on in my analysis.
colnames(meas_subset)
#I have a total of 28 measures here, varing from vital measurements from a patient encounter to metrics about a social environment. Let's look at what is the type of variables I can see from these 
# measures though
meas_subset[,3]
#Column 3 was body temperature, and though we appear to have some null values from what I'm seeing from the small set of them, they all seem to hoover around
# what is a normal body temp. which is 98 degrees
meas_subset[,24]
#Column 25 is the measure pertaining to the percentage of uninsured for the county the patient comes from. These are generally low numbers which 
# is a good thing for our society, but if our patient has a higher probability to fall into these unemployment numbers that can be an issue for their health.
#There are also variables which are binary that determine if the patient is from a high risk area or not for a specific societal metric, let's see one of those.
meas_subset[,18]
#Here column 18 is the Family Income Tract, which determines if the Median family income for an area is above or below the national average, this has no 
# barring on the patient themselves, just where they come from based on their county 

#That is a good look into the types of measures were going to see within our dataset but what do the measures look like in a single row?
meas_subset[c(1, 3, 5, 7,9),]
#Definitely curious to look at this, to see if I can see anything that sticks out, nothing really but there are so many measures that it is hard for just one to stick out.

#Looking through both the dimensional and measurable variables it is easy to see I can still take out some more variables before I get deeper into my analysis.

#Since ICD_Group is one of my more important dimension variables, I'd like to take a look into it and see the kinds of insight I can find out about each of its factors.
#Before doing so, I'm going to omit a few more variables so I can declutter my dataset for better analysis
str(bpp)
bpp_final <- select(bpp, ICD_Group, CONTACT_DATE, CITY, State, ZIP, County, SEX_C, ETHNIC_GROUP_C, MARITAL_STATUS_C, LANGUAGE_C, TOBACCO_USE_VRFY_YN, 
                    BIRTH_DATE, BP_SYSTOLIC, BP_DIASTOLIC, TEMPERATURE, PULSE, WEIGHT, RESPIRATIONS, BMI, BSA,Crime_Rate, Urban,
                    Poverty_Rate, Poverty_Tract, Median_Family_Income, MFI_Tract, Access_to_Supermarket, Distant_from_Supermarket, 
                    High_School_Graduation_Rate, HS_Graduation_Tract, Percentage_Unemployed, Unemployed_Tract, Eligible_Free_Reduced_Lunch, Free_Reduced_Lunch_Tract,
                    Uninsured_Adults, Uninsured_Tract, Food_Insecurity, Food_Insecurity_Tract, Food_Environmental_Index, FEI_Tract, Frequent_Mental_Distress, Mental_Distress_Tract,
                    Physical_Inactivity, Physical_Inactivity_Tract, Access_to_Exercise, Access_to_Exercise_Tract, Flu_Vaccinations, Excessive_Drinking, Excessive_Drinking_Tract,
                    Flu_Vaccinations_Tract, Mammography, Mammography_Tract)

head(bpp_final)
#I now have a mix of my dimensions and measures that will be ideal for analysis going forward
str(bpp_final)
summary(bpp_final)
#Our bpp_final dataset still has quite a few variables but it has cut down a substantial 51 variables in total 

bpp_final[c(1,3,5,7,9),]
#It is still hard to see the entire data set unfortunately, but we have all the variables we would want to use moving forward so let's look deeper into understanding our dataset.

#We can start by looking at some charts which will help us understand the distribution of the data.
cc_dist <- table(bpp_final$ICD_Group)
cc_dist
barplot(cc_dist, col = "blue")
#Our chronic conditions aren't distributed evenly but there is a good amount of each, obesity is by far the most, then hypertension, and finally diabetes.

state_dist <- table(bpp_final$State)
state_dist
barplot(state_dist, col = "red")
#As anticipated, Colorado is where a majority of our patients reside. 
city_dist <- table(bpp_final$CITY)
city_dist
barplot(city_dist, col = "black")
#The spread on our cities is much more varied, which isn't a surprise because we have a lot of different cities within our patient population
#What is surprising to me isn't the that places like Denver and Arvada are some of the higher areas were seeing places, but places like Topeka, KS
# have such a high population of patients residing from that specific part of the country.

county_dist <- table(bpp_final$County)
county_dist
barplot(county_dist)
#No real surprise here, seeing Jefferson far and away the highest populated county from our patient population is what is expected, same with Denver
# county being up there as well. I'd like to look a little bit more into the different variables though because those are going to be how we understand what 
# variables are the biggest indicator for a specific disease or not.

with(bpp_final, {plot(BP_SYSTOLIC~BP_DIASTOLIC)
  lines(lowess(BP_SYSTOLIC~BP_DIASTOLIC), col = "red")})
#There is definitely a positive correlation between the two blood pressures, but that is expected.
with(bpp_final, {plot(TEMPERATURE~PULSE)
  lines(lowess(TEMPERATURE~PULSE), col = "red")})
#Surprisingly to me, it appears that our patients tend to have the same general body temperature, which makes it difficult to judge what kind of 
# correlation that could have to any variables, that's disappointing, let's look at pulse with another variable
with(bpp_final, {plot(PULSE~RESPIRATIONS)
     lines(lowess(PULSE~RESPIRATIONS), col= "red")})
#Again, two variables to find difficult to find a trend to, the variances are different so it is harder to notice any trend
#It might be easier to understand our data more if we look at the distributions of our variables see what we should be expecting and the best way to do
# this is by looking at histograms
##Histograms
#BP_Systolic
hist(bpp_final$BP_SYSTOLIC, breaks = 10)

#BP_Diastolic
hist(bpp_final$BP_DIASTOLIC, breaks = 10)
#The distribution of these two variables are almost carbon copies of one another, only difference is systolic has higher numbers to diastolic

#Temperature
hist(bpp_final$TEMPERATURE, breaks = 10)
#This is showing again that the range in temperatures is quite stuck within that normal 98 degree area

#Pulse
hist(bpp_final$PULSE, breaks = 10)
#The pulse of these patients appear to in a range that is a little higher than 60 beats/min

#Weight
hist(bpp_final$WEIGHT, breaks = 10)
#The distribution says its in the 1000s but its really in the 100s and the majority of the patients look to be over 200 lbs

#Respirations
hist(bpp_final$RESPIRATIONS, breaks = 10)
#Respirations look to all be in the same general area, just under 20.

#BMI
hist(bpp_final$BMI, breaks = 10)
#Each of these BMI's appear to be on the high side of this measure typically, this doesn't mean that our population is necessarily oveweight but 
# we can deduce that is likely

#BSA
hist(bpp_final$BSA, breaks = 10)
#Quite the normal distribution, no skew in any real direction.

#Poverty Rate
hist(bpp$Poverty_Rate, breaks = 10)
#Poverty rate skewed quite a bit but plenty of records with a rate above 20.

#Median Family Income
hist(bpp$Median_Family_Income, breaks = 10)
#Our patient population appears to come from households that are just under $100K for median earnings, which is higher than expected 
# but this doesn't mean our patients are at this level its just based on where they live not actually what they make.

#SuperMarket Access
hist(bpp_final$Access_to_Supermarket, breaks = 10)
#It appears for the most part our patients have good access to a super market

#Access to Exercise
hist(bpp_final$Access_to_Exercise, breaks = 10)
#It would appear our patients have no issues being able to exercise within their environment

#High School graduation rate
hist(bpp_final$High_School_Graduation_Rate, breaks = 10)
#Much of the population is around the national average of 84.6%

#Unemployment rate
hist(bpp_final$Percentage_Unemployed, breaks = 10)
#Unemployment around the country is very low, that is why our patients appear to come from low unemployment areas, but that doesn't mean
# this is true for the individual patient

#Uninsured patients
hist(bpp_final$Uninsured_Adults, breaks = 10)
#Uninsured rate in America falls just over 8%, It would seem we have a lot of patients fall under that, but there is probably a lot hovering 
# around or right above it, we would need a more refined chart to truly know.

#Free Reduced Lunch Rate
hist(bpp_final$Eligible_Free_Reduced_Lunch, breaks = 10)
#The majority of our patients look to be in areas where the lunch rate is above 40% of the population

#Food Insecurity
hist(bpp_final$Food_Environmental_Index, breaks = 10)
#The food insecurity index of our patients is skewed on the high end between 8 and 10 %.

#Frequent Mental Distress
hist(bpp_final$Frequent_Mental_Distress, breaks = 10)
#It apppears our patient population is huddled right by the 11.1% threshold of average in the USA

#Physical Inactivity
hist(bpp_final$Physical_Inactivity, breaks = 10)
#Physical Inactivity rates are scattered throughout the distribution, no area more  populated than between 15 - 20 %

#Excessive Drinking
hist(bpp_final$Excessive_Drinking, breaks = 10)
#Excessive drinking looks to be an issue for a majority of our patients where it looks like 20% of adults partake in excessive drinking.

#Flu Vaccinations
hist(bpp_final$Flu_Vaccinations, breaks = 10)
#Skewed high, it appears that the flu vaccinations most patients come from areas where the flu vaccination is taken by at least 45% of the population

#Mammography
hist(bpp_final$Mammography, breaks = 10)
#Appears the mammography rate in this area is higher than 40% for most of the patient population

#With the rest of then numbers being binary, I will move onto looking at these variables when compared across dimensions. The one in 
# particular that I have a lot of interest in is the ICD_Group because it houses what type of chronic condition the record is.
#For these comparisons I will start by looking at box plots

##Medical Measures - Vitals
#BP_Diastolic
boxplot(bpp_final$BP_DIASTOLIC ~ bpp_final$ICD_Group, ylab = "Blood Pressure Diastolic",
        xlab = "Chronic Conditions", main = "BP Diastolic vs. Chronic Condition", cex.lab=1.5)
#BP_Systolic
boxplot(bpp_final$BP_SYSTOLIC ~ bpp_final$ICD_Group, ylab = "Blood Pressure - Systolic",
        col = rep(c(1,2,3)),xlab = "Chronic Conditions", main = "BP Systolic vs. Chronic Condition", 
        cex.lab = 1.5, cex.axis = 1.5, cex.main = 2)

#This does a great job showing the differences within the different types of patients we see per chronic condition. Appears the blood pressure for each chronic condition
# is similar but diabetes does appear to be on the higher end than both obesity and hypertension but not enough to state as significant.

#Temperature
boxplot(bpp_final$TEMPERATURE ~ bpp_final$ICD_Group, ylab = "Body Temp.",
        xlab = "Chronic Conditions", main = "Body Temp. vs. Chronic Conditions", col = rep(c(1,2,3)), 
        cex.lab = 1.5, cex.axis = 1.5, cex.main =2)
#Body temperatures appear to be fairly similar, obesity looks to be higher than diabetes & hypertension though

#Pulse
boxplot(bpp_final$PULSE ~ bpp_final$ICD_Group, ylab = "Pulse",
        xlab = "Chronic Conditions", main = "Pulse vs. Chronic Conditions", col = rep(c(1,2,3)),
        cex.lab = 1.5, cex.axis = 1.5, cex.main= 2)
#The pulse of each are quite similar, the medians appear to be just below 80 with some high outliers around or at 120 for some patients w/ each CC

#Weight
#Before doing the box plot for weight, I'd like to change this measure because it is multipled by 10 for some reason and makes it seem like each patient
# weighs over 1000 lbs. 
bpp_final$WEIGHT <- bpp_final$WEIGHT /10
head(bpp_final$WEIGHT)
#Now that the weights look good, let's make the box plot
boxplot(bpp_final$WEIGHT ~ bpp_final$ICD_Group, ylab = "Weight", xlab = "Chronic Conditions",
        main = "Weight vs. Chronic Conditions", col = rep(c(1, 2, 3, 4, 5, 6, 7, 8, 9)),
        cex.lab = 1.5, cex.axis = 1.5, cex.main = 2)
#No surprise here that diabetes & obesity have the highest patients in terms of weight, some outliers w/ hypertension but their quartile range is much lower than 
# that of diabetes & obesity.

#Respirations
boxplot(bpp_final$RESPIRATIONS ~ bpp_final$ICD_Group, ylab = "Respirations", xlab = "Chronic Conditions",
        main = "Respirations vs. Chronic Conditions", col = rep(c(1, 2, 3, 4, 5, 6, 7, 8, 9)),
        cex.lab = 1.5, cex.axis = 1.5, cex.main =2)
#All the resipiration ranges look about the same, with a couple of crazy outliers around 120 that look like mistakes in the records.

#BMI
boxplot(bpp_final$BMI ~ bpp_final$ICD_Group, ylab = "BMI", xlab = "Chronic Conditions",
        main = "BMI vs. Chronic Conditions", col = rep(c(1,2,3,4,5,6,7,8,9)),
        cex.lab = 1.5, cex.axis = 1.5, cex.main =2)
#The BMI ranges are highest for obesity, but diabetes is a distinct 2nd and than hypertension as 3rd as expected.

#BSA
boxplot(bpp_final$BSA ~ bpp_final$ICD_Group, ylab = "BSA", xlab = "Chronic Conditions",
        main = "BSA vs. Chronic Conditions", col = rep(c(1,2,3,4,5,6,7,8,9)),
        cex.lab = 1.5, cex.axis = 1.5, cex.main =2)

#BSA gave us the same results as the BMI, which means there is probably a strong correlation between the 2.

#Those box plots were great for understanding if there was any difference in the distribution of the individual variables, but it doesn't give 
# any indication on any correlating variables possible, just what might be different from one chronic condition to another. Let's check some of the social 
#determinants for these correlations.

#Next, I want to look at qplots of some pairs of social determinants

#Poverty Rate vs. Median Family Incom
qplot(Poverty_Rate, Median_Family_Income, data = bpp_final, facets = .~ ICD_Group) + geom_smooth()
#The 3 CC's appear to be a bit similar and not showing a whole lot of trending based on the CC and the MFI and Poverty Rates

#MFI vs. Supermarket Access
qplot(Median_Family_Income, Low_Access_to_Supermarket, data = bpp_final, facets = .~ ICD_Group) + geom_smooth()
#The cc's appear to trend similarily here with this pair of Social determinants as well, no distinct trend between the two variables.

#High School Grad Rate vs. Percentage Unemployed
qplot(High_School_Graduation_Rate, Percentage_Unemployed, data = bpp_final, facets = .~ ICD_Group) + geom_smooth()
#This was interesting because all 3 CC's are contrasted quite a bit by the trendline but that is not noticeable by seeing the datapoints

#Free/Reduced Lunch vs. Uninsured Adults
qplot(Eligible_Free_Reduced_Lunch, Uninsured_Adults, data = bpp_final, facets = .~ ICD_Group) + geom_smooth()
#Each CC has a positive trend by the pair of variables, none of the lines are the same though, they are different based on how their data points 
# are plotted 

#Food Insecurity vs. Food Environment Index
qplot(Food_Insecurity, Food_Environmental_Index, data = bpp_final,facets = .~ ICD_Group) + geom_smooth()
#The data points noticeably trend down, as the food insecurity rises, the FEI drops and this is consistent across each CC.

#Frequent Mental Distress vs. Physical Inactivity
qplot(Physical_Inactivity, Frequent_Mental_Distress, data = bpp_final, facets = .~ ICD_Group) + geom_smooth()
#These variables all cause the data trend differently based on the CC. Only hypertension trends positively, but the start of each line is quite similar

#Excessive Drinking vs. Flu Vaccinations
qplot(Excessive_Drinking, Flu_Vaccinations, data = bpp_final, facets = .~ICD_Group) + geom_smooth()
#Based on the cc's the data trends similarly with its parabolic line. The data points for each are quite sporadic which makes sense for the data points.

#The qplots are a great way to show the differences within dimensions across different variable pairs. 

#################################################################################################################

##Beginning the Cluster Analysis

#Before I get into my HCA and K-means clustering I need to change the look of my dataset one last time. I will need to make it 
# so I only have the one variable I'm clustering on and then all my different measures. The variable I want to 
# cluster upon to start is ICD_Group (Chronic Condition) because I want to be able to see how different these are from one another.
colnames(bpp_final)

bpp_final1 <- select(bpp_final, ICD_Group, BP_SYSTOLIC, BP_DIASTOLIC, TEMPERATURE, PULSE, WEIGHT, RESPIRATIONS, BMI, BSA,
                     Poverty_Rate, Median_Family_Income, Access_to_Supermarket, High_School_Graduation_Rate, Percentage_Unemployed,
                      Eligible_Free_Reduced_Lunch, Uninsured_Adults, Food_Insecurity, Food_Environmental_Index, Frequent_Mental_Distress,
                     Physical_Inactivity, Access_to_Exercise, Excessive_Drinking, Flu_Vaccinations, Mammography)
#This leaves us with just the chronic condition labels and 23 different variables to look into and see what kind of hidden patterns can be possibly pulled from it.

head(bpp_final1)

str(bpp_final1)

#Before looking into the cluster analysis, I want to look at a chart which will help me see the correlation of my variables.
# A correlogram will help me compare and measure the correlation of each measure.
install.packages("ggcorrplot")
library(ggcorrplot)

#Get rid of null values
bpp_final1 <- na.omit(bpp_final1)
nrow(bpp_final1)
#With losing the null values we had to drop 68 records.

cc_corr <- select(bpp_final1, -ICD_Group)
#I need to remove the last variable that isn't numeric from my data
corr <- round(cor(cc_corr), 1)

ggcorrplot(corr, hc.order = TRUE,
           type = "lower",
           lab = TRUE,
           lab_size = 3,
           method = "circle",
           colors = c("tomato2", "white","springgreen3"),
           title = "Correlogram of Chronic Conditions",
           ggtheme = theme_bw)

#To validate what were seeing from the correlogram, we can do the same thing but with a correlation matrix with pie charts:
install.packages("corrplot")
library(corrplot)
corrplot(corr, "pie","lower")

#The correlograms help us distinguish the positive and negative correlations throughout our 23 variables. The positive correlations appear to be 
# between: BP_Systolic vs. BP_Diastolic, BMI vs. Weight, BMI vs. BSA, BSA, vs. Weight, Uninsured Adults vs. Free Reduced Lunch Rate,
# Food Environmental Index, Frequent Mental Distress vs. Free Reduced Lunch Rate, Frequent Mental Distress vs. Uninsured Adults, Frequent Mental Distress 
# vs. Food Insecurity, Physical Inactivity vs. High School Grad Rate, Access to Exercise vs. Median Family Income, Access to Exercise vs. Food Environmental Index
# Excessive Drinking vs. Median Family Income, Execessive Drinking vs. Access to Exercise, Flu Vaccinations vs. MFI, Flu Vaccinations vs. Access to Exercise
# Mammography Rate vs. Flu Vaccinations
#The negative correlations are Free Reduced Lunch Rate vs MFI, FRLR vs High School Grad Rate, Uninsured Adults vs. Median Family Income
#Food Insecurity vs. Median Family Income, Food Environmental Index vs. Food Insecurity, Frequent Mental Distress vs. Median Family Income, Frequent Mental Index vs. FEI,
#Physical Inactivity vs. Median Family Income, Physical Inactivity vs. FEI, Access to Exercise vs. High School Grad Rate, Access to Exercise vs. Physical Inactivity,
# Excessive drinking vs. High School Grad rate, Excessive Drinking vs. Physical Inactivity, Flu Vaccinations vs Physical Inactivity, Mammography Rate vs. Uninsured Adults.

#It is great to see an indication of what variables correlate before doing the cluster analysis because it helps with understanding how the 
# data points should plot when looking at these data points 

colnames(bpp_final1)
#It looks like I will have a total of 17 columns going into my cluster analysis, with ICD_Group and 16 numeric variables.
str(bpp_final1)
#There are two types of data types, ICD_Group is the only non-numeric variable, and it is of type factor.
summary(bpp_final1)
#This is a great visualization and will help us with understanding the variables which might help us understand the mapping of our HCA.

###HCA

library(colorspace)
head(bpp_final1)
bpp2 <- bpp_final1[,-1]
str(bpp2)
bpp2 <- as.data.frame(bpp2)
#It might more conducive to normalize our data before we use it. so let's do that as well before going through our HCA & K-means clustering
bpp2 <- normalize(bpp2, method = "standardize", range = c(0,1), margin = 1L, on.constant = "quiet")
bpp2$BP_SYSTOLIC <- ((bpp2$BP_SYSTOLIC - min(bpp2$BP_SYSTOLIC)) / (max(bpp2$BP_SYSTOLIC) - min(bpp2$BP_SYSTOLIC)))
bpp2$BP_DIASTOLIC <- ((bpp2$BP_DIASTOLIC - min(bpp2$BP_DIASTOLIC)) / (max(bpp2$BP_DIASTOLIC) - min(bpp2$BP_DIASTOLIC)))
bpp2$TEMPERATURE <- ((bpp2$TEMPERATURE - min(bpp2$TEMPERATURE)) / (max(bpp2$TEMPERATURE) - min(bpp2$TEMPERATURE)))
bpp2$PULSE <- ((bpp2$PULSE - min(bpp2$PULSE)) / (max(bpp2$PULSE) - min(bpp2$PULSE)))
bpp2$WEIGHT <- ((bpp2$WEIGHT - min(bpp2$WEIGHT)) / (max(bpp2$WEIGHT) - min(bpp2$WEIGHT)))
bpp2$RESPIRATIONS <- ((bpp2$RESPIRATIONS - min(bpp2$RESPIRATIONS)) / (max(bpp2$RESPIRATIONS) - min(bpp2$RESPIRATIONS)))
bpp2$BMI <- ((bpp2$BMI - min(bpp2$BMI)) / (max(bpp2$BMI) - min(bpp2$BMI)))
bpp2$BSA <- ((bpp2$BSA - min(bpp2$BSA)) / (max(bpp2$BSA) - min(bpp2$BSA)))
bpp2$Poverty_Rate <- ((bpp2$Poverty_Rate - min(bpp2$Poverty_Rate)) / (max(bpp2$Poverty_Rate) - min(bpp2$Poverty_Rate)))
bpp2$Median_Family_Income <- ((bpp2$Median_Family_Income - min(bpp2$Median_Family_Income)) / (max(bpp2$Median_Family_Income) - min(bpp$Median_Family_Income)))
bpp2$Access_to_Supermarket <- ((bpp2$Access_to_Supermarket - min(bpp2$Access_to_Supermarket)) / (max(bpp2$Access_to_Supermarket) - min(bpp2$Access_to_Supermarket)))
bpp2$Food_Environmental_Index <- (bpp2$Food_Environmental_Index - min(bpp2$Food_Environmental_Index)) / (max(bpp2$Food_Environmental_Index) - min(bpp2$Food_Environmental_Index))
head(bpp2)

#With the data normalized, we have every variable on the same scale which is will make it easy to see if there is any differences between them.
#
chronic_conditions_labels <- bpp_final1[,1]
chronic_conditions_labels <- unlist(chronic_conditions_labels)
cc_color <- rev(rainbow_hcl(10))[as.numeric(chronic_conditions_labels)]

pairs(bpp2, col = cc_color,
      lower.panel = NULL,
      cex.labels = 1, pch=19, cex= 1.2)

par(xpd = TRUE)
legend("topleft", x = 0.025, y = 0.2, cex = 2,
legend = as.character(levels(chronic_conditions_labels)),
fill = unique(cc_color))
par(xpd = NA)

#It is hard to find any distinction between any of the variables and chronic conditions. There are 23 variables & 3 Chronic conditions that can happen
# but what we can see is some of the correlations between variables highlighted from the correlogram again in this pairs plot.

#We can see conclusions as well with a parallel coordinates plot of the data:
par(las = 1, mar = c(4.5,3,3,2) + 0.1, cex =0.8)
MASS::parcoord(bpp2, col = cc_color, var.label = TRUE, lwd= 2)
#Add title
title("Parellel coordinate plot of Chronic Conditions")
#Legend
par(xpd = TRUE)
legend(x =1.75, y = -.25, cex = 1,
       legend = as.character(levels(chronic_conditions_labels)),
       fill = unique(cc_color), horiz = TRUE)
par(xpd = NA)
#There isn't much distinction between any of the data points & the different variables based on the CC's assigned. 

##HCA will help us now look at these chronic conditions and see how to cluster these based on these variables
dist_cc <- dist(bpp2)
hc_cc <- hclust(dist_cc, method = "complete")
cc_types <- rev(levels(bpp_final1[,1]))

library(dendextend)
dend <- as.dendrogram(hc_cc)
#order it by the order of the observations:
dend <- rotate(dend, 1:1049)

#Color the branches based on the clusters:
dend <- color_branches(dend, k=3, groupLabels = cc_types)
#This will group by the chronic condition

#The bpp_final1 needs to be unlisted so it can be used as sort level values
unlist(bpp_final1[,1])

unlist(order.dendrogram(dend))
#Manually match the labels, to the real classification of the chronic conditions:
labels_colors(dend) <-
  rainbow_hcl(3)[sort_levels_values(
    as.numeric(bpp_final1[,1])[order.dendrogram(dend)]
  )]

#Add the Chronic condition to the labels:
labels(dend) <- paste(as.character(bpp_final1[,1])[order.dendrogram(dend)],
                      "(",labels(dend),")",
                      sep = "")
#We hang the dendrogram a bit:
dend <- hang.dendrogram(dend, hang_height = 0.1)
#Reduce the size of the labels:
# dend <- assign_values_to_leaves_nodePar(dend, 0.5, "lab.cex")
dend <- set(dend, "labels_cex", 0.5)
#Plot it:
par(mar = c(1,1,2,1))
plot(dend,
     main = "Clustered Chronic Conditions",
     horiz = TRUE, nodePar = list(cex = 0.007))
legend("left", 
       x = 0.025, y = 0.2, cex = 1,
       legend = as.character(levels(chronic_conditions_labels)),
       fill = unique(cc_color))
par(xpd = NA)
#It appears that we do have 3 distinct clusters here, blue is hypertension, obesity is red and green is diabetes. 

#Let's look at this now in a circular form to see if it will be just as easy to spot the groupings 
install.packages("circlize")
library(circlize)
par(mar = rep(0,4))
circlize_dendrogram(dend)
#The same result is shown by the circular dendrogram. The three types of CC's are grouped togther 
# almost perfectly.

#Now let's see what adding the effects of a heat map can do for finding how the data clusters
install.packages("gplots")
library(gplots)
some_col_func <- function(n) rev(colorspace:: heat_hcl(n, c = c(80, 30), l = c(30,90),
                                                       power= c(1/5, 1.5)))

gplots:: heatmap.2(as.matrix(bpp2),
                   main = "Heatmap for the Chronic Conditions",
                   strCol = 20,
                   dendrogram = "row",
                   Rowv = dend,
                   Colv = "NA", #makes sure the columns are not ordered
                   trace = "none",
                   margins = c(5, 0.1),
                   key.xlab = "Measure Value",
                   denscol = "grey",
                   density.info = "density",
                   #RowSideColors = rev(labels_colors(dend)), #adds the colored strips
                   col = some_col_func
)

#From the heat map of the individual variables, there is no apparent trend between the variables and the chronic conditions, but with so many variables
# that is hard to distinguish through a heat map.


#Let's show the entire variable group together to prove the scales are off. 
d3heatmap::d3heatmap(as.matrix(bpp2),
                     dendrogram = "row",
                     Rowv = dend,
                     colors = "Greens",
                     width = 600,
                     show_grid = FALSE)
#Thankfully the data is normalized so we can see each variable again on a 0-1 scale but again it doesn't show much cause there isn't much 
# of a pattern with the data here.

#We need to ask ourselves if this was the best method of HCA for our data, there are 8 different algorithm methods 
# that can be possibly implemented so we can test them against one another.
hclust_methods <- c("ward.D", "single","complete","average", "mcquitty","median", "centroid", "ward.D2")
cc_dendlist <- dendlist()
for(i in seq_along(hclust_methods)) {
  hc_cc <- hclust(dist_cc, method = hclust_methods[i])
  cc_dendlist <- dendlist(cc_dendlist, as.dendrogram(hc_cc))
}
names(cc_dendlist) <- hclust_methods
cc_dendlist
#The method I used originally was the complete method, which we are seeing is actually our lowest based on height.
#We can create a correlation of these as well and see which methods relate more to each other. 

#Comparing ward.D and ward.D2 methods 
cc_dendlist %>% dendlist(which = c(1, 8)) %>% ladderize %>%
  set("branches_k_color", k = 3) %>%
  tanglegram(faster = TRUE)
#Comparing these 2 methods, there is quite a difference. These variables have a lot of crossing between them based on the 2 methods.

#Comparing all 8 methods
par(mfrow = c(4,2))
for(i in 1:8) {
  cc_dendlist[[i]] %>% set("branches_k_color", k = 3) %>% plot(axes = FALSE, horiz = TRUE)
  title(names(cc_dendlist)[i])
}
#The 8 methods do give quite different results, the best results come from ward.D, complete and ward.D2. The others don't appear
# to show any true difference between the chronic conditions.

#Test the correlation of the 8 types
cc_dendlist_cor <- cor.dendlist(cc_dendlist, method = "common")
cc_dendlist_cor
#Plot it
corrplot::corrplot(cc_dendlist_cor, "pie", "lower")
#It appears from charting and plotting the different types, that our different methods are comparable all across the board, 
#no two more than ward.D and ward.D2. With how close these are though, they're pretty similar in comparison.
