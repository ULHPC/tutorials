library(tidyverse)
library(furrr)
library(Rtsne)

# check the path of pkgs
.libPaths()
# see how many cores are going to be used
availableCores()
# load data
tSNEdata <- readRDS("tSNEdata.rds")
# create a partial function with filled arguments 
short_tsne <- purrr::partial(Rtsne, X = tSNEdata, pca_center = TRUE, 
                            theta = 0.5, pca_scale = TRUE, verbose = TRUE, max_iter = 300)

#plan(multisession) # via sockets
plan(multiprocess)
plan(multicore) # like multiprocess, full processes

tictoc::tic(msg = "tsne")
tsne_future <- tibble(perplexities = seq(10, 110, by = 5)) %>%
                 # quietly captures outputs
                 mutate(model = future_map(perplexities, ~quietly(short_tsne)(perplexity = .x), .progress = FALSE))
tictoc::toc()


tictoc::tic("finding clusters")
res_tib <- mutate(tsne_future,
                  # unlist and extract the 2D matrix 
                  Y = map(model, pluck, "result", "Y"),
                  # convert to a dataframe
                  Y_df = map(Y, as.data.frame),
                  # for clustering, parallelise since expensive step
                  cluster = future_map(Y_df, dbscan::hdbscan, minPts = 200),
                  # extract from hdbscan object only the cluster info
                  c = map(cluster, pluck, 1),
                  # iterate though the 2D coordinates and cluster info to merge them
                  tsne = map2(Y_df, c, ~ bind_cols(.x, tibble(c = .y))),
                  # extract the output of tsne runs if needed to be parsed
                  output = map_chr(model, pluck, "output"))
tictoc::toc()

saveRDS(res_tib, "tsne_future.rds")

