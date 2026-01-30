# Data Sources

## 1. Injury Episodes
- **Source**: Kaggle
- **Dataset**: `xfkzujqjvx97n/football-datasets` (Transfermarkt-derived)
- **Description**: Injury episodes with `player_id`, start/end dates, and days missed. Used as the primary label source.

## 2. Appearances/Minutes
- **Source**: Kaggle
- **Dataset**: `davidcariboo/player-scores`
- **Description**: Match-level player statistics, games, and player info.

## 3. Market Value / Transfers
- **Source**: Kaggle
- **Dataset**: `xfkzujqjvx97n/football-datasets`
- **Description**: Market value time series + transfer history used as context features.

## 4. Alternative Injury Sources (Optional)
- **Source**: Kaggle
- **Dataset**: `irrazional/transfermarkt-injuries`
- **Description**: Backup injury dataset (also includes `player_id`). Useful for coverage checks.

## 5. Player Attributes (Optional / Not Joinable)
- **Source**: Kaggle
- **Dataset**: `kolambekalpesh/football-player-injury-data`
- **Description**: Physical attributes (height, weight, etc.), but does not share the same `player_id` system as the main pipeline.
