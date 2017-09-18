# Project 1

install.packages('Lahman')
library(Lahman)


# a)
# We  calculate the sum of "stolen bases" and the sum of "at bat" for each 
# team since year 2000. We name these as BatSB_Sum and BatAB_Sum.
BatSB_Sum <- sapply(split(Batting$SB[Batting$yearID >= 2000], 
                          Batting$teamID[Batting$yearID >= 2000]), 
                    sum, na.rm = TRUE)
BatAB_Sum <- sapply(split(Batting$AB[Batting$yearID >= 2000], 
                          Batting$teamID[Batting$yearID >= 2000]), 
                    sum, na.rm = TRUE)

# We divide BatSB_Sum vector by BatAB_Sum vector to obtain Stolen bases per
# at bat ratio (SBperAB). However, if BatAB_Sum is zero, we replace the 
# value by NA.
SBperAB <- ifelse(BatAB_Sum != 0, BatSB_Sum/BatAB_Sum, NA)
# SBperAB is sorted and the sorted values and theit labels are 
# turned into vectors.
SBperAB_sorted <- as.vector(sort(SBperAB, decreasing = TRUE))
SBperAB_sorted_lb <- as.vector(labels(sort(SBperAB, decreasing = TRUE)))

# a data frame shows the first 5 teams and their respective SBperAB values.
data.frame(TeamID = head(SBperAB_sorted_lb)[-6], 
           SBperAB = head(SBperAB_sorted)[-6])


# *********************************************

# b)
# We split Master$birthYear by itself to obtain a list containing 
# the birth years, each year repeated to the number of players who were born
# in that year. Then the number of each repetition is counted using length().
players_year <- sapply(split(Master$birthYear, Master$birthYear), length)

# The name of the year in which the maximum number of players were born is
# shown.
labels(players_year[players_year==max(players_year)])

# We split Master$birthMonth by itself to obtain a list containing 
# the birth months, each month repeated to the number of players who were born
# in that month. Then the number of each repetition is counted using length().
players_month <- sapply(split(Master$birthMonth, Master$birthMonth), length)

# the name of the month in which the maximum number of players were born is
# shown using subsetting the builtin vector month.name
month.name[as.numeric(labels(players_month[players_month==max(players_month)]))]


# *********************************************

# c)
# We merge the needed columns from Batting and Master by playerID and name
# the result as MasBatHR.
MasBatHR <- merge(Master[,c('playerID', 'nameFirst', 'nameLast')], 
      Batting[,c('playerID', 'HR')])

# We split MasBatHR$HR by playerID to obtain the homerun by each player and then
# add them using sum (via sapply).
PlayerHR <- sapply(split(MasBatHR$HR, MasBatHR$playerID), sum, na.rm = TRUE)

# Here we find the playerID of the player with maximum homeruns
maxID <- labels(PlayerHR[PlayerHR = max(PlayerHR)])
# We obtain the first name and last name of that player and turn them
# into characters.
f <- as.character(MasBatHR[MasBatHR$playerID == maxID, 'nameFirst'])
l <- as.character(MasBatHR[MasBatHR$playerID == maxID, 'nameLast'])
 # We join the first and last name together and print it.
FullNameMaxHR <- paste(f,l)
print(FullNameMaxHR)


# *********************************************

# d)
# The needed columns of Salaries and Master data frames are merged 
# (by playerID) and the result is named SalMas.
SalMas <- merge(Salaries[,c('yearID', 'teamID','playerID','salary')],
                Master[,c('playerID', 'nameFirst', 'nameLast')])

# A two-way split (by teamID and yearID) is made on SalMas$salary, and 
# then the sum of salaries is calculated using sapply. This results in a
# vector containing the sum of salaries paid by each team in each year. This
# is called SalMassplt.
SalMassplt <- sapply(split(SalMas$salary, list(SalMas$teamID, SalMas$yearID)),
                     sum, na.rm = TRUE)

# The label of the component of SalMassplt for which the salary is maximum 
# contains the teamID and yearID that is needed. We extract this info as
# follows:
maxpaying <- labels(SalMassplt[SalMassplt == max(SalMassplt)])
teamyear <- strsplit(as.character(maxpaying),'[.]')
team <- teamyear[[1]][1]; team
year <- teamyear[[1]][2]; year
 
# Using the teamID and yearID that was found, we subset SalMas to obtain 
# player names and their respective salaries. The result is named SalmasMax.  
SalmasMax <- SalMas[SalMas$teamID == team & SalMas$yearID == year, 
                    c('nameFirst', 'nameLast', 'salary')]
# We rearrange SalmasMax in the order of decreasing salaries and list the names.
SalmasMax[order(SalmasMax$salary, decreasing = TRUE), c('nameFirst', 'nameLast', 
                                                        'salary')]


# *********************************************

# e)
# We  calculate the total "stolen bases" and "at bat" for each 
# year, summed over all teams. We name these as BatSB_Sum_yr and BatAB_Sum_yr.
BatSB_Sum_yr <- sapply(split(Batting$SB, 
                          Batting$yearID), 
                    sum, na.rm = TRUE)
BatAB_Sum_yr <- sapply(split(Batting$AB, 
                          Batting$yearID), 
                    sum, na.rm = TRUE)

# We divide BatSB_Sum_yr vector by BatAB_Sum_yr vector to obtain the yearly 
# Stolen bases per at bat ratio (SBperAB_yr). However, if BatAB_Sum_yr is 
# zero, we replace the value by NA. The result is named SBperAB_yr.
SBperAB_yr <- ifelse(BatAB_Sum_yr != 0, BatSB_Sum_yr/BatAB_Sum_yr, NA)

# A numeric vector, containing years, is made from the labels of SBAByear.
years <- as.numeric(labels(SBperAB_yr))

# SBperAB_yr is plotted against years.
plot(years, SBperAB_yr, xlab = 'Year' , ylab = 'yearly SB.Per.AB')


# *********************************************

# f)
# We put SBperAB_yr and years together in a data frame to facilitate
# our work
SBAB_yr <- data.frame(SBperAB_yr=SBperAB_yr, years = years)

# We find the interquartile range and by using the quantile
# function calculate the lower bound and upper bound (lb and ub) 
# of the data would not contain the outliers. 
iqr <- IQR(SBperAB_yr)
lb <- as.numeric(quantile(SBperAB_yr)[2] - 1.5*iqr)
ub <- as.numeric(quantile(SBperAB_yr)[4] + 1.5*iqr)

# We subset the data points to leave out the outliers
SBAB_yr_sub <- subset(SBAB_yr, SBAB_yr$SBperAB_yr >= 
                        lb & SBAB_yr$SBperAB_yr <= ub)

# plotting the original data and the data without outliers
plot(SBAB_yr$years, SBAB_yr$SBperAB_yr,
     xlab = 'Year' , ylab = 'yearly SB.Per.AB')
plot(SBAB_yr_sub$years, SBAB_yr_sub$SBperAB_yr,
     xlab = 'Year' , ylab = 'yearly SB.Per.AB')

# The 2nd way of way of cleaning the data: leaving out years before 1887 
YA <- SBAB_yr$years[SBAB_yr$years > 1886]
SA <- SBAB_yr$SBperAB_yr[SBAB_yr$years > 1886]
plot(YA, SA,  xlab = 'Year' , ylab = 'yearly SB.Per.AB')

# When we look at the plot of the original data, it seems that there are 
# 16 data points before year 1887 which do not fall nicely within the general
# trend that is observable after that year. On the other hand, when we use 
# lb = Q1 - 1.5*iqr and ub = Q3 + 1.5*iqr (as above) to filter the data, 
# what gets eliminated is not these 16 bad points which do not follow the 
# general trend, but the data points between year 1887 and 1900 which form 
# a reasonable part of the general trend in the beginning of it. So, because 
# of these reasons I decided to abandon the cleaning by using lb and ub and
# go with the cleaning by just eliminating the first 16 data points.

# Applying linear regression to the uncleaned data.
lmSBAB_yr_uncleaned <- lm(SBperAB_yr ~ years)
summary(lmSBAB_yr_uncleaned)
plot(SBperAB_yr ~ years,  xlab = 'Year' , ylab = 'yearly SB.Per.AB')
lines(years, predict(lmSBAB_yr_uncleaned), col = "red")
# The line is underfitting the visible trend. 

# Applying linear regression to the data cleaned in the 2nd way
lmSBAB_yr <- lm(SA ~ YA)
summary(lmSBAB_yr)
plot(SA~YA,  xlab = 'Year' , ylab = 'yearly SB.Per.AB')
lines(YA, predict(lmSBAB_yr), col = "red")
# The line is still underfitting the visible trend.

# Applying polynomial regression to the uncleaned data.
polSBAB_yr_uncleaned <- lm(SBperAB_yr ~ years + I(years^2)+
                             I(years^3)+I(years^4)+I(years^5))
summary(polSBAB_yr_uncleaned)
plot(SBperAB_yr~years,  xlab = 'Year' , ylab = 'yearly SB.Per.AB')
lines(years, predict(polSBAB_yr_uncleaned), col = "red")
# The initial data points are diverting the fitted curve from 
# following the clear trend seen in the data after 1886.

# Applying polynomial regression to the data cleaned in the 2nd way
polSBAB_yr <- lm(SA ~ YA + I(YA^2)+I(YA^3)+I(YA^4)+I(YA^5))
summary(polSBAB_yr)
plot(SA~YA,  xlab = 'Year' , ylab = 'yearly SB.Per.AB')
lines(YA, predict(polSBAB_yr), col = "red")
# The fitted curve is following the major trend visible in the data nicely. 
# So, we adopt this as our predictive model.
# The reason we are using a 5th order polynomial to fit the data is that
# a polynomial up to 4th order does not capture the trend as good as a 5th
# order one, and then increasing the order beyond 5 would not make a
# significant improvement. So we adopt the lower order (5th) which captures
# the trend well enough.

# *********************************************

# g)
# player contains the playerID and all the years per playerID
player <- split(Appearances[,c('yearID', 'playerID')], Appearances$playerID)
# This line removes any NA entries from the player list.
player <- player[!is.na(player)]

# unip is a vector containing all the unique years that a player has played
unip <- unique(player[[1]][[1]])

# The 1st way (whithout using if or loop)
activeYears <- function(v) {
  v_sh <- c(tail(v, -1), head(v, 1)) # shifts the v to the right by one
  diff <-  v_sh - v # finds the difference between successive years in v
  wdiff <- which(diff[1:length(diff)-1] != 1) # finds the positions where the difference 
  # between successive years is not 1 (except the last one which is the subtraction of the
  # last and the first element).
  wdiff <- c(0,wdiff,length(v)) # pads wdiff by 0 from left and length(v) from right 
  shwdiff <- wdiff[c(2:length(wdiff))] # shifts wdiff by one to the left (excluding the 1st element)
  col <- shwdiff - wdiff[1:length(wdiff)-1] # this gives the lengths of continuous playing
  col
}


# The 2nd way
activeYears2 <- function(v) {
  # This loop will produce a vector named col2 which will contain 
  # the sequence of the lengths of continuous playing 
  flow <- NULL # flow will be used as a temporary dump for the years
  col2 <- NULL # col2 will contain the the sequence of the lengths of continuous playing
  for (i in 1:length(v)){
    if (i == 1 && length(v) > 1){
      ct <- 1
      flow <- c(flow, v[i])
      cf <- 1
      next
    } else if (i == 1 && length(v) == 1) {
      col2 <- 1
      break
    }
    flow <- c(flow, v[i])
    cf <- cf + 1
    if(flow[cf] == (flow[cf-1]+1) && i < length(v)){
      ct <- ct + 1
      #cf <- cf + 1
    } else if (flow[cf] == (flow[cf-1]+1) && i == length(v)){
      ct <- ct + 1
      col2 <- c(col2, ct)
    }
    if (flow[cf] != (flow[cf-1]+1) && i < length(v)) {
      col2 <- c(col2, ct)
      flow <- v[i]
      cf <- 1
      ct <- 1
    } else if (flow[cf] != (flow[cf-1]+1) && i == length(v)){
      flow <- v[i]
      col2 <- c(col2,ct)
      col2 <- c(col2, 1)
    }
  }
  col2
}

unip
activeYears(unip) # The result of the 1st way
activeYears2(unip) # The result of the 2nd way


