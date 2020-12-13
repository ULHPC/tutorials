

#' wrap download.file to return the filename
download_file <- function(url, out) {
  download.file(url, out, quiet = TRUE)
  out
}