library(segmented)

# filename = './urml_mean_ellip_result.csv'
m2mm = 1000.0
g_const = 9.806

# dat <- read.csv(filename)
dat$dis <- dat$dis*m2mm
dat$force <- dat$force/g_const
dat$tacc <- dat$tacc/g_const

# dy <- 0.00167662*m2mm
# du <- 0.0067065*m2mm

#####################

#####################

tf <- dat$dis > dy
ndat <- data.frame(x=log(dat$dis[tf]),y=log(dat$tacc[tf]-dat$force[tf]))
ndat <- ndat[complete.cases(ndat),]

lin <- lm(y~x, ndat)
#seg <- segmented(lin, seg.Z = ~x, psi=c(1.5,2.0))
seg <- segmented(lin, seg.Z = ~x, psi=2.0)

dev.new()
part_new <- data.frame(x=log(dat$dis))
part_new$y <- predict(seg, part_new)
part_new$y_fit <- exp(part_new$y) + dat$force 
plot(dat$dis, part_new$y_fit - dat$tacc, col='red')

#####################

#####################

ndat1 <- data.frame(x=log(dat$dis),y=log(dat$tacc/dat$force)**0.5)
ndat1 <- ndat1[complete.cases(ndat1),]

lin1 <- lm(y~x, ndat1)
#seg1 <- segmented(lin1, seg.Z = ~x, psi=c(1.5,2.0))
seg1 <- segmented(lin1, seg.Z = ~x, psi=1.5)

dev.new()
plot(y~x, ndat1)
points(ndat1$x, seg1$fitted, col='red')

part_new$y1 <- predict(seg1, part_new)
part_new$y1_fit <- exp(part_new$y1**2.0)*dat$force

dev.new()
plot(tacc~dis, dat)
points(dat$dis, part_new$y1_fit, col='red')

dev.new()
plot(dat$dis, part_new$y1_fit - dat$tacc, col='red')

#####################

#####################

ndat2 <- data.frame(x=log(dat$dis), y = log(dat$tacc/dat$force))

lin2 <- lm(y~x, ndat2)
seg2 <- segmented(lin2, seg.Z=~x, psi=c(2.0,3.0))

part_new$y2 <- predict(seg2, part_new)
part_new$y2_fit <- exp(part_new$y2)*dat$force

dev.new()
plot(dat$dis, part_new$y2_fit - dat$tacc, col='red')

#####################

#####################

ndat3 <- data.frame(x=log(dat$dis), y=log(dat$tacc))

dev.new();plot(y~x,ndat3)

lin3 <- lm(y~x, ndat3)
seg3 <- segmented(lin3, seg.Z=~x, psi=c(2,4))

part_new$y3 <- predict(seg3, part_new)
part_new$y3_fit <- exp(part_new$y3)

dev.new()
plot(dat$dis, part_new$y3_fit - dat$tacc, col='red')

#####################

#####################

tf <- dat$dis > dy

ndat4 <- data.frame(x=log(dat$dis[tf]),y=log(dat$tacc[tf]/dat$force[tf])**0.5)
ndat4 <- ndat4[complete.cases(ndat4),]

lin4 <- lm(y~x, ndat4)
seg4 <- segmented(lin4, seg.Z = ~x, psi=2.5)

part_new <- data.frame(x=log(dat$dis))
part_new$y4 <- predict(seg4, part_new)
part_new$y4_fit <- exp(part_new$y4**2.0)*dat$force

plot(dat$dis[!tf], part_new$y4_fit - dat$tacc, col='red')



#####################

#####################

ndat5 <- data.frame(x=log(dat$dis),y=log(dat$tacc/0.6)**0.5)
ndat5 <- ndat5[complete.cases(ndat5),]

lin5 <- lm(y~x, ndat5)
#seg1 <- segmented(lin1, seg.Z = ~x, psi=c(1.5,2.0))
#seg1 <- segmented(lin1, seg.Z = ~x, psi=1.5)

dev.new()
plot(y~x, ndat5)
points(ndat5$x, lin5$fitted, col='red')

part_new$y5 <- predict(lin5, part_new)
part_new$y5_fit <- exp(part_new$y5**2.0)*0.6

dev.new()
plot(tacc~dis, dat)
points(dat$dis, part_new$y1_fit, col='red')

dev.new()
plot(dat$dis, part_new$y1_fit - dat$tacc, col='red')


# model comparison
# less AIC better?
# Not sure
# 