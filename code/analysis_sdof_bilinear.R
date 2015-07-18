#!/usr/bin/Rscript
require(segmented)

args <- commandArgs(TRUE)
filename = args[1]
threshold = as.numeric(args[2])

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

m2mm = 1000.0
g_const = 9.806

dat <- read.csv(filename)

dat$dis <- dat$dis*m2mm
threshold <- threshold*m2mm
dat$force <- dat$force/g_const
dat$tacc <- dat$tacc/g_const

tf <- dat$dis > threshold

# unit conversion mm and g
ndat <- data.frame(x=log(dat$dis[tf]), y=log(dat$tacc[tf]-dat$force[tf]))
ndat <- ndat[complete.cases(ndat),]
lin.mod <- lm(y~x, ndat)
seg <- segmented(lin.mod, seg.Z = ~x, psi=log(1.1*threshold))
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

fitted_seg <- function(seg, ndat) {

	intercept <- seg$coefficients[1]
	slope1 <- seg$coefficients[2]
	slope2 <- seg$coefficients[3]+seg$coefficients[2]
	break_pt <- seg$psi[2]

	y <- numeric(length(ndat$x))

	# tf = dat$dis < dy
	# abs_acc[tf] = dat$force[tf]
	# print(sum(tf))

	tf = ndat$x < break_pt
    y[tf] = intercept + slope1*ndat$x[tf]

    tf = ndat$x >= break_pt
    y[tf] = intercept + slope1*break_pt + slope2*(ndat$x[tf]-break_pt)

	return(y)
}

