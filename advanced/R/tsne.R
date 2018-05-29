library(tidyverse)
library(furrr)
library(Rtsne)

availableCores()
# 4 on macbook pro
tSNEdata <- readRDS("tSNEdata.rds")
arno_tsne <- partial(Rtsne, X = tSNEdata, pca_center = TRUE, 
                            theta = 0.5, pca_scale = TRUE, verbose = TRUE, max_iter = 300)



plan(multisession)
tictoc::tic()
system.time(tibble(perplexities = seq(10, 110, by = 5)) %>%
  mutate(model = future_map(perplexities, ~quietly(arno_tsne)(perplexity = .x), .progress = TRUE)) -> tsne_future)

tictoc::toc()

