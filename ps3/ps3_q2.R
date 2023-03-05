library(matrixStats)

# Function to get position of n-th largest value
# From https://stackoverflow.com/questions/10296866/finding-the-column-number-and-value-the-of-second-highest-value-in-a-row
maxn <- function(n) function(x) order(x, decreasing = TRUE)[n]

# Function to compute (negative) likelihood for 2.6 (i)
# params[1] is mu, params[2] is sd, sorted_dataframe is the sorted dataframe
likelihood_12 <- function(params, sorted_dataframe) {
  likelihood <- data.frame(matrix(ncol = 1, nrow = 1000))
  likelihood[, 1] <- log(dnorm(sorted_dataframe[, 10], mean = params[1], sd = params[2])
                         /(1 - pnorm(sorted_dataframe[, 9], mean = params[1], sd = params[2])))
  return(-mean(likelihood[, 1]))
}

# Function to compute (negative) likelihood for 2.6 (ii)
# params[1] is mu, params[2] is sd, sorted_dataframe is the sorted dataframe
likelihood_13 <- function(params, sorted_dataframe) {
  likelihood <- data.frame(matrix(ncol = 1, nrow = nrow(sorted_dataframe)))
  likelihood[, 1] <- log((2*(pnorm(sorted_dataframe[, 10], mean = params[1], sd = params[2]) 
                             - pnorm(sorted_dataframe[, 8], mean = params[1], sd = params[2]))
                          *dnorm(sorted_dataframe[, 10], mean = params[1], sd = params[2]))
                         /((1 - pnorm(sorted_dataframe[, 8], mean = params[1], sd = params[2]))^2))
  return(-mean(likelihood[, 1]))
}

# Set random number generation for replication
set.seed(29)

# Set up the main dataframe
df <- as.data.frame(matrix(rnorm(n = 10000), nrow = 1000, ncol = 10))
names(df) <- c(1:10)

# Sort it in ascending order
df_sorted <- t(apply(df, 1, sort))
df_sorted

# Problem 2.2
mean(df_sorted[, 10])
mean(df_sorted[, 9])

### eBay style auction

# Set up empty data frame containing bids
ebay_bids <- data.frame(matrix(ncol = 10, nrow = 1000))
names(ebay_bids) <- c(1:10)

# eBay bids for first two bidders are just their valuations
ebay_bids[, 1] <- df[, 1]
ebay_bids[, 2] <- df[, 2]

# Make new data frame to store second highest bids. This is the minimum bid faced
# by each bidder from i=3, ..., 10
second_high <- data.frame(matrix(ncol = 8, nrow = 1000))
names(second_high) <- c(3:10)

for(i in 3:10) {
  # Find the second highest bid so far
  second_high[, (i-2)] <- apply(ebay_bids[, 1:(i-1)], 1, function(x)x[maxn(2)(x)])
  
  # Implement bidding strategy
  ebay_bids[, i] <- ifelse(df[, i] >= second_high[, (i-2)], df[, i], NA)
}

# Which bidders actually submit
bidder_identity <- data.frame(matrix(ncol = 10, nrow = 1000))
names(bidder_identity) <- c(1:10)
bidder_identity[ , ] <- ifelse(is.na(ebay_bids[ , ]), 0, 1)

# Create number of bidders per auction
num_bidders <- apply(bidder_identity, 1, sum)

# Problem 2.4
summary(num_bidders)
sd(num_bidders)

# Make new data set containing mean and variance of mean
mean_var <- data.frame(matrix(ncol = 5, nrow = 1000))
names(mean_var) <- c('Mean', 'Var', 'tval', 'pval', 'Significant')
mean_var[, 'Mean'] <- rowMeans(ebay_bids, na.rm = TRUE)
mean_var[, 'Var'] <- rowVars(data.matrix(ebay_bids), na.rm = TRUE) / num_bidders[]

# Calculate t-value and p-value for each auction (testing against null of 0)
mean_var[, 'tval'] <- mean_var[, 'Mean'] / sqrt(mean_var[, 'Var'])
mean_var[, 'pval'] <- 2*pt(-abs(mean_var[, 'tval']), df=num_bidders-1)

# Find if it's significant at different levels
mean_var[, 'Significant_1'] <- ifelse(mean_var[, 'pval'] < 0.01, 1, 0)
mean_var[, 'Significant_5'] <- ifelse(mean_var[, 'pval'] < 0.05, 1, 0)
mean_var[, 'Significant_10'] <- ifelse(mean_var[, 'pval'] < 0.10, 1, 0)

# Problem 2.5
mean(mean_var[, 'Significant_1'])
mean(mean_var[, 'Significant_5'])
mean(mean_var[, 'Significant_10'])

# Make a vector of all bids and then test if the mean is equal to 0
output <- unlist(ebay_bids, use.names = FALSE)
tval <- mean(output, na.rm = TRUE) / sqrt(var(output, na.rm = TRUE) / sum(!is.na(output)))
pval <- 2*pt(-abs(tval), df = sum(!is.na(output)) - 1)
tval 
pval

# Get first and second highest bids
ebay_sorted <- ebay_bids
ebay_sorted[is.na(ebay_sorted)] <- -100
ebay_sorted <- t(apply(ebay_sorted, 1, sort))
ebay_sorted <- as.data.frame(ebay_sorted)

# Find the MLE estimates for 2.6 (i)
optim(c(3, 5), likelihood_12, sorted_dataframe = ebay_sorted, method = 'Nelder-Mead')

# Now for first-third highest we have to drop observations without 3 bidders
ebay_sorted <- subset(ebay_sorted, V8 > -100)

# Find the MLE estimates for 2.6 (ii)
optim(c(3, 5), likelihood_13, sorted_dataframe = ebay_sorted, method = 'Nelder-Mead')