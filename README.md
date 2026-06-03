# Chess Rating Prediction via Player Network Embeddings

This project predicts chess player ratings using Lichess match data, graph-based player relationships, Node2Vec embeddings, and XGBoost regression.

Instead of only using basic win-loss statistics, the project treats players as nodes in a competition network. Match outcomes are modeled as directed edges between players, allowing the model to capture opponent strength, matchup history, and hidden similarities between players.

## Project Overview

The goal of this project is to estimate official chess player ratings from match-level and network-based features.

The project combines:

- Python data processing
- Graph-based feature engineering
- Node2Vec player embeddings
- XGBoost regression
- RMSE model evaluation
- Feature importance analysis

## Tools Used

- Python
- Pandas
- NumPy
- NetworkX
- Node2Vec
- XGBoost
- Scikit-learn
- Matplotlib

## Features Engineered

The model uses several graph and match-based features, including:

- PageRank
- Betweenness Centrality
- K-Core decomposition
- Win/loss relationships
- Opponent strength indicators
- Node2Vec embeddings
- Match-level performance patterns

## Model

An XGBoost regression model was trained to predict player ratings using the engineered network features and player embeddings.

Model performance was evaluated using RMSE, and feature importance was used to identify which network patterns were most connected to chess skill.

## Why This Project Matters

This project shows how graph-based machine learning can reveal patterns that basic statistics may miss. By modeling chess players as part of a competitive network, the model can learn from the quality of opponents, player relationships, and hidden similarities across the player pool.

## Resume Summary

Built a chess rating prediction model using Lichess match data, Node2Vec network embeddings, graph-based features, and XGBoost regression to predict official player ratings.
