library(votesmart)
rm(list=ls())

token <- read.delim("token.txt")
token <- token[1, 1]
Sys.setenv(VOTESMART_API_KEY = token)
key <- Sys.getenv("VOTESMART_API_KEY")

key_exists <- (nchar(key) > 0)
if(!key_exists) knitr::knit_exit()
suppressPackageStartupMessages(library(dplyr))
conflicted::conflict_prefer("filter", "dpylr")

# lnames = read.csv("candidates.csv")
# 
# for(lname in lnames$last_name) {
#   fpath = paste(paste("cands/",lname,sep=""),".csv", sep="")
#   if(!file.exists(fpath)) {
#     (tmp <-
#         votesmart::candidates_get_by_lastname(
#           last_names = lname,
#           election_year = 1980:2020
#         )
#     )
#     tryCatch({
#       tmp <- tmp %>% filter((!is.na(candidate_id))) %>% select(
#         candidate_id, first_name, last_name, office_type_id,
#         election_year, election_state_id, election_office
#       )
#       write.csv(tmp, fpath)
#     },
#     error=function(cond){
#       print("Error encountered")
#       print(tmp)
#     })
#   }
# }

ids = read.csv("cand_ids.csv")
ids = ids$cand_id %>% unique()
# print(ids)

for(id in ids) {
  fpath = paste(paste("sigs/",id,sep=""),".csv", sep="")
  if(!file.exists(fpath)) {
    (tmp <-
       votesmart::rating_get_candidate_ratings(
         candidate_ids = id
       ) 
    )
    tryCatch({
      tmp <- tmp %>% select(-c("rating_text"))
      write.csv(tmp, fpath)
    },
    error=function(cond){
      # print(cond)
      print("Error encountered")
      print(tmp)
    })
  }
}

(congressional_cand_id <-
    congressional_cand %>%
    pull(candidate_id) %>%
    unique()
)

(cong_ratings <-
    rating_get_candidate_ratings(
      candidate_ids = congressional_cand_id,
      sig_ids = ""
    )
)
cong_ratings <- cong_ratings %>% select(
  -c("category_id_6","category_id_7","category_id_8",
     "category_name_6","category_name_7","category_name_8")
)
write.csv(cong_ratings, "cong_ratings.csv")

