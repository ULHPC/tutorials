

#' wrap download.file to return the filename
download_file <- function(url, out) {
  download.file(url, out, quiet = TRUE)
  out
}

#' linear model on country, liefExp explained by year1950
#' so we get meaningful intercept
ml_lifeExp <- function(.data) {
  gp <- mutate(.data, year1950 = year - 1950)
  lm(lifeExp ~ year1950, data = gp)
}


#' extract r.square from lm object
extract_r2 <- function(model) {
  summary(model)$r.squared
}