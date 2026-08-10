# FAL Destroyer 3000

## Goals
1. Predict end MAL score for roster selection (Model A)
2. Use time series forecasting to maximize points in the Fantasy Anime League (Model B)

## ETL Phase

### Data Collection

Uses Tenrai API unless said otherwise

Model A Data:
* Thumbnail
* Title
* Source
* Episode Count
* Synopsis (for semantic analysis)
* Season and Year
* Studios
* Producers
* Age Rating
* Genres
* Demographics
* Themes
* Is it a sequel? 

Model A Prediction Variables (which will then be used as Model B's priors)
* Score
* Watching + Completed (z-score against cohort)
* Total forum messages for first thirteen episodes (z-score against cohort) (STRONG assumption: linear correlation between forum messages and unique users)
* Dropped (z-score against cohort)
* Favorites (z-score against cohort)

## Commit Notes
* Drafted extraction function
* Extraction function does not have a failsafe for breaking 24-hour rate limits. Will need a way to save the csv regardless, remind the user of what page number did the breaking occur, and a way to "continue" the extraction phase when rerun