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

## Interesting Findings from EDA
* Higher score averages for the mid 90's and lower score averages for the mid 2010's.
* Slightly higher score averages for shows under the restriction of "Violence & Profanity."
* Slightly higher score averages for sequel shows.

## Feature Engineering
* Combined "season" and "year" into "cohort" like "Spring 2025"
* Turned wc, favorites, forum, and score into z-scores against cohort
* Turned dropped --> drop rate (dropped / wc) --> z-score of drop rate against cohort
* Turned every single image whose MAL ID is in the CSV into a padded 448x448 image to prepare for wd-eva02
* Performed semantic analysis on all synopses and did PCA, reducing components to 193
* Performed bucketing for genres and themes
* Performed multi-layer binarization of genres, themes, and demographics
* Decided that studios and producers have pretty small overlap and correlation. Planning to make a separate learned embedding

## Commit Notes
* Did a good chunk of EDA, will most likely continue with visualizing relationships between metrics.
* Two new .py files: adaptation_collection and stats_collection. The former will collect the scores and member count of all possible source materials (manga, LN, etc) from one anime. This will be z-scored against works of the same source material type. In the end, if there are multiple sources, the highest z-score will be chosen-- this will serve as our "source material hype factor" that could be a really good indicator for success. The latter will collect all possible scores and member counts from all possible sources that can be accessible. The mean and standard deviation will help determine the z-score for the aforementioned adaptation metrics.
* adaptation_collection.py has not been tested yet because stats_collection.py is still ongoing.
* image_extraction saves all thumbnails
* Was not able to do AniList GraphQL API because it's highly prone to mismatched titles