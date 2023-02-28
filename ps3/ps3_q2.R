library(matrixStats)

# Function to get position of n-th largest value
# From https://stackoverflow.com/questions/10296866/finding-the-column-number-and-value-the-of-second-highest-value-in-a-row
maxn <- function(n) function(x) order(x, decreasing = TRUE)[n]

# Set random number generation for replication
set.seed(29)

# Set up the main dataframe
df <- as.data.frame(matrix(rnorm(n = 10000), nrow = 1000, ncol = 10))
names(df) <- c(1:10)

# Sort it in ascending order
df_sorted <- t(apply(df, 1, sort))

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

# Problem 2.3
summary(num_bidders)
sd(num_bidders)

# Make new data set containing mean and var of bids
mean_var <- data.frame(matrix(ncol = 2, nrow = 1000))
names(mean_var) <- c('Mean', 'Var')
mean_var[, 'Mean'] <- rowMeans(ebay_bids, na.rm = TRUE)
mean_var[, 'Var'] <- rowVars(data.matrix(ebay_bids), na.rm = TRUE)