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

## Data Cleaning

The following modifications were done:
* Removed anime with no episode count as they are still ongoing and thus metrics would fluctuate more than if the series was already finished.
* Removed anime with no scores as they tend to be on the more obscure and regionally-broadcasted side. This was partially confirmed by checking the other metrics of scorelessa nime.
* Removed anime with no age rating as I believe that anime with no age rating could mess with analysis. More restrictive anime could have slightly worse metrics.
* Stripped the endings of synopses that contain messages like "Written by MAL Rewrite" or "Synopsis written by..." as they can affect sentiment analysis.

## Commit Notes
* Failsafe done
* Need one more day to finish data collection
* Already drafted a data cleaning notebook