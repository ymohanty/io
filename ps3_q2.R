# Set random number generation for replication
set.seed(29)

# Set up the main dataframe
df <- as.data.frame(matrix(rnorm(n=10000), nrow=1000, ncol=10))
names(df) <- c('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10')

# Sort it in ascending order
df_sorted <- t(apply(df,1,sort))

# Mean of highest bid and second highest bid
mean(df_sorted[,10])
mean(df_sorted[,9])