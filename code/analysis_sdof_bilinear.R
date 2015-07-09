#!/usr/bin/Rscript
require(segmented)

args <- commandArgs(TRUE)
filename = args[1]
period = as.numeric(args[2])

sink(paste(strsplit(filename,'.csv'),'.txt',sep=""))

set.seed(123)

print(filename)

#filename = '/Users/hyeuk/Project/PCEE2015/data/urml_mean_bin_result_damp05.csv'
#period = 0.15

# gmotion <- read.csv('/Users/hyeuk/Project/sv01-30/sv_psa.csv', header=FALSE)
# gm_period <- as.numeric(gmotion[1,])
# idx_period <- which(gm_period==period)
# gm_psa <- gmotion[-1,idx_period]

# dat <- read.csv(filename)
# dat$psa <- 0

# for (i in seq(1,30)) {
#     tf = dat$gm==sprintf(fmt="sv%02d",i)
#     dat$psa[tf] <- dat$scale[tf]*gm_psa[i]
# }

dat <- read.csv(filename)
omega_2 <- (2.0*pi/period)**2.0
dy <- max(dat$force)/omega_2
tf <- dat$dis > dy

# unit conversion mm and g
ndat <- data.frame(x=log(dat$dis[tf]*1000.0), y=log(dat$tacc[tf]-dat$force[tf])-log(9.806))
ndat <- ndat[complete.cases(ndat),]
lin.mod <- lm(y~x, ndat)
seg <- segmented(lin.mod, seg.Z = ~x, psi=log(1.1*dy*1000.0))
summary(seg)
slope(seg)

edat <- data.frame(x=dat$force[!tf], y=dat$tacc[!tf])
lin.e <- lm(y~x, edat)
summary(lin.e)
sink()

jpeg(file=paste(strsplit(filename,'.csv'),'_el.jpeg',sep=""))
plot(y~x, edat)
points(edat$x, fitted(lin.e),col=2, pch=20)
#plot(seg, add=TRUE, col='red')
dev.off()

jpeg(file=paste(strsplit(filename,'.csv'),'.jpeg',sep=""))
plot(y~x, ndat)
points(ndat$x,broken.line(seg,link=FALSE)$fit,col=2,pch=20)
#plot(seg, add=TRUE, col='red')
dev.off()

jpeg(file=paste(strsplit(filename,'.csv'),'_resid.jpeg',sep=""))
plot(ndat$x, resid(seg), col='red')
dev.off()

