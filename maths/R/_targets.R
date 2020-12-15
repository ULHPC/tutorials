library(targets)
library(tarchetypes)
source("R/functions.R")
options(tidyverse.quiet = TRUE)
# necessary packages to load in isolated R sessions
tar_option_set(packages = c("tidyverse", "fst"))
# for debugging, activate the following, a workspace will be created in
# _targets/workspaces/failed_object. Where `failed_object` is the name of failure
# Load with tar_workspace(failed_object)
# when done, remove with tar_destroy(destroy = "workspace")
#tar_option_set(error = "workspace")

#library(future)
#plan(multicore)

# Define the pipeline
tar_pipeline(
  tar_url(gp_url, "https://raw.githubusercontent.com/jennybc/gapminder/master/inst/extdata/gapminder.tsv"),
  tar_file(gp_file, download_file(gp_url, "gapminder.tsv")),
  tar_fst_tbl(gp, read_tsv(gp_file, col_types = cols())),
  tar_fst_tbl(gp_grp,
              group_by(gp, continent, country) %>%
                tar_group(),
              # tell downstream targets about the grouping
              iteration = "group"),
  tar_target(models,
             ml_lifeExp(gp_grp),
             pattern = map(gp_grp),
             # lm is complex, combine in a list
             iteration = "list"),
  tar_target(r2,
             extract_r2(models),
             pattern = map(models),
             # now vector is enough
             iteration = "vector"),
  tar_render(report, "report.Rmd")
)
