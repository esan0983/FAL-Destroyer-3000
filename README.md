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
* Popularity of source material (score + members)

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

## Statistics

We will ignore the Award Winning genre because of their obvious high metrics.

### Score
* Twelve genres were shown to have a statistically significant effect. In particular, the Sports and Drama genres has the biggest effect. The Ecchi and Horror genres have a big negative effect on score.
* More than half of the themes had a statistically significant effect. In particular, the "Iyashikei" (slice-of-life) theme had the biggest positive effect. The themes Strategy Game, Parody, and Harem had pretty big negative effects.
* All demographics had a statistically significant effect. Only the Kids demographic had a negative effect.

### Watching + Completed
* Only three genres did not have a statistically significant effect. The Romance and Suspense genres had the biggest positive effect.
* More than half of the themes had a statistically significant effect. Love Polygon, Gore, and Isekai were three of the most postively impactful themes, while Pets had the biggest negative impact.
* All demographics had a statistically significant effect. The Kids demographic had a really big negative effect.

### Favorites
* Only two genres did not have a statistically significant effect. The Romance and Suspense genres had the biggest positive ffect. The horror genre had the biggest negative effect.
* More than half of the themes had a statistically significant effect. Love Polygon was the most positively impactful theme, while Pets and Strategy Game had the biggest negative impact.
* All demographics had a statistically significant effect. The Kids demographic had a really big negative effect.

### Forum Posts
* Only two genres did not have a statistically significant effect. The Romance and Suspense genres had the biggest positive ffect. The horror genre had the biggest negative effect.
* More than half of the themes had a statistically significant effect. Love Polygon and Psychological were the most positively impactful themes, while Pets and Strategy Game had the biggest negative impact.
* All demographics had a statistically significant effect. The Kids demographic had a really big negative effect.

### Overall Effects
* Taking rating and the sequel boolean, the top 3 most positive statistically significant features on score (barring award winning) are: Iyashikei Theme, Shounen Demographic, and Gag Humor Theme. The top 3 most negative statistically significant features on score are: Horror Genre, Kids Demographic, and Ecchi Genre.
* The top 3 most positive statistically significant features on score (barring award winning) are: R17 Violence & Profanity Rating, Otaku Culture Theme, and Shounen Demographic. The top 3 most negative statistically significant features on score are: Kids Demographic, Strategy Game Theme, and Samurai Theme.
* The top 3 most positive statistically significant features on score (barring award winning) are: R17 Violence & Profanity Rating, Iyashikei Theme, and Shounen Demographic. The top 3 most negative statistically significant features on score are: Strategy Game Theme, Horror Genre, and Samurai Theme.
* The top 3 most positive statistically significant features on score (barring award winning) are: R17 Violence & Profanity Rating (actually bigger than award winning), PG13 Age Rating, and R17 Mild Nudity Rating. The top 3 most negative statistically significant features on score are: Strategy Game Theme, Kids Demographic, and PG Children Rating.

## Limitations
* Was not able to do AniList GraphQL API because it's highly prone to mismatched titles

## Commit Notes
* Redid get_anime util function, etl2.py, and image_extraction.py. Have not tested yet due to rate limits
* Improved data cleaning and EDA section
* Performed log1p wc, favorites, and forum first before z-scoring due to heavy data skew
* Collected source material statistics to perform z-scoring to evaluate source material hype
* Collected prequel statistics
* Collected embedding tensors for both thumbnail and synopsis
* Integrated most external data collection and data cleaning protocols to etl2.py

## Post-commit Plans
* Will run adaptation_collection first and do a test run of the ML notebook before revising and redoing the entire pipeline from etl2.py all the way to the ML notebook
* Will add bools to the dataset such as "has_prequel" and "has_source"