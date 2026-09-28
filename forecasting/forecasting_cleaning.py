import pandas as pd
import duckdb

from utils import parse_list_col

def clean(forecasting_df, initial_df, week) -> None:
    print(f"Week {week}...")
    initial_df['title'] = initial_df['title'].str.strip()
    forecasting_df['anime'] = forecasting_df['anime'].str.strip()

    combined_df = duckdb.sql("""
        select
            f.anime,
            i.mal_id,
            i.source,
            i.cohort,
            i.genres,
            i.demographics,
            i.themes,
            i.rating,
            i.score,
            i.wc,
            i.prequel_id,
            i.doujinshi_score,
            i.doujinshi_members,
            i.manhua_score,
            i.manhua_members,
            i.manhwa_score,
            i.manhwa_members,
            i.novel_score,
            i.novel_members,
            i.light_novel_score,
            i.light_novel_members,
            i.manga_score,
            i.manga_members,
            i.one_shot_score,
            i.one_shot_members,
            f.points
        from forecasting_df f
        left join initial_df i on f.anime = i.title
    """).df()

    # count how many mismatched titles there are
    lost_num = duckdb.sql("""
        select count(*)
        from combined_df
        where mal_id is null
    """).fetchone()[0]

    print(f"Removing {lost_num}/{len(combined_df)} rows...")
    combined_df = combined_df.dropna(subset=['mal_id'])

    print("Doing the rest of the cleaning...")

    combined_df['points'] = combined_df['points'].astype(str).str.replace(',', '').str.strip()
    combined_df['points'] = pd.to_numeric(combined_df['points'], errors='coerce')
    combined_df = combined_df.drop_duplicates(subset=['anime'], keep='first')

    lists = ['genres', 'demographics', 'themes']
    combined_df[lists] = combined_df[lists].map(parse_list_col)

    combined_df.to_parquet(f"data/forecasting/processed/week{week}_processed.parquet")
    print(f"Parquet saved for week {week}!")

def main() -> None:
    initial_df = pd.read_csv("data/raw/anime_data.csv")
    for i in range(13):
        forecasting_df = pd.read_csv(f"data/forecasting/raw/week{i+1}.csv", encoding_errors="replace")
        clean(forecasting_df, initial_df, i+1)

if __name__ == "__main__":
    main()

    
