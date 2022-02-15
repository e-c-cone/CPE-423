library(votesmart)
#library(comprehenr)

#token <- read.fwf("token.txt", header=FALSE)
#token <- read.delim("token.txt")
Sys.setenv(VOTESMART_API_KEY = "3b67e66f0dc7c1482a2317dbff292cef")
key <- Sys.getenv("VOTESMART_API_KEY")
print(token)
print(key)

key_exists <- (nchar(key) > 0)
if(!key_exists) knitr::knit_exit()
suppressPackageStartupMessages(library(dplyr))
conflicted::conflict_prefer("filter", "dpylr")

(cand <-
    votesmart::candidates_get_by_lastname(
      last_names = "grijalva",
      election_year = 1980:2020
    )
)

(barneys <-
    cand %>%
      filter(first_name != "") %>%
      select(
        candidate_id, first_name, last_name,
        election_year, election_state_id, election_office
      )
)

(barney_id <-
    barneys %>%
    pull(candidate_id) %>%
    unique()
)

(barney_ratings <-
    rating_get_candidate_ratings(
      candidate_ids = barney_id,
      sig_ids = ""
    )
)

