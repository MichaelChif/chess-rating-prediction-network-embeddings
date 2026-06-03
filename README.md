# Chess Rating Prediction via Player Network Embeddings

This project predicts chess player ratings using Lichess match data, graph-based player relationships, Node2Vec embeddings, and XGBoost regression.

Instead of only using basic win-loss statistics, the project treats players as nodes in a competition network. Match outcomes are modeled as directed edges between players, allowing the model to capture opponent strength, matchup history, and hidden similarities between players.

## Project Overview

The goal of this project is to estimate official chess player ratings from match-level and network-based features.

The project combines:

* Python data processing
* Graph-based feature engineering
* Node2Vec player embeddings
* XGBoost regression
* RMSE model evaluation
* Feature importance analysis
* Interactive network visualizations

## Why This Project Matters

Chess ratings are usually understood through individual performance, but player strength is also shaped by the quality of opponents and the structure of competition. A player who wins against stronger opponents should be represented differently from a player who wins against weaker opponents.

This project uses graph-based machine learning to model those relationships. By representing players as nodes and match outcomes as edges, the model can learn patterns that basic statistics may miss.

## Features

The project includes the following steps:

* Parse Lichess match data
* Filter games into a usable format
* Build a directed player competition graph
* Model wins as directed edges between players
* Engineer graph-based features
* Generate Node2Vec player embeddings
* Train an XGBoost regression model
* Evaluate performance using RMSE
* Analyze feature importance
* Create static and interactive visualizations

## Tech Stack

* Python
* Pandas
* NumPy
* NetworkX
* Node2Vec
* XGBoost
* Scikit-learn
* Matplotlib
* HTML visualizations

## Features Engineered

The model uses several graph-based and match-level features, including:

* PageRank
* Betweenness Centrality
* K-Core decomposition
* Win and loss relationships
* Opponent strength indicators
* Node2Vec embeddings
* Match-level performance patterns

## Model

An XGBoost regression model was trained to predict official chess player ratings using the engineered network features and player embeddings.

Model performance was evaluated using RMSE, and feature importance was used to understand which network patterns were most connected to chess skill.

## Project Structure

```text
chess-rating-prediction-network-embeddings/
├── app/
│   ├── __init__.py
│   ├── build_graph.py
│   ├── config.py
│   ├── features.py
│   ├── parse_pgn.py
│   ├── run_pipeline.py
│   ├── train.py
│   ├── utils.py
│   ├── visualize_embeddings.py
│   └── visualize_interactive_network.py
├── data/
│   ├── games_filtered.parquet
│   ├── graph_edges.parquet
│   └── player_features.parquet
├── models/
│   ├── player_graph.pkl
│   └── xgb_rating_model.joblib
├── outputs/
│   ├── feature_importance.png
│   ├── interactive_network.html
│   ├── interactive_network.png
│   ├── metrics.json
│   ├── node2vec_embedding_map.html
│   ├── node2vec_embedding_map.png
│   └── predictions.csv
├── README.md
└── requirements.txt
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/MichaelChif/chess-rating-prediction-network-embeddings.git
cd chess-rating-prediction-network-embeddings
```

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline

```bash
python -m app.run_pipeline
```

This runs the main project pipeline, including data processing, graph construction, feature engineering, model training, and output generation.

## Outputs

Generated files are saved in the `outputs/` folder.

The outputs may include:

* Model performance metrics
* Rating predictions
* Feature importance chart
* Interactive player network visualization
* Node2Vec embedding visualization

## Example Outputs

### Feature Importance

The feature importance chart shows which graph-based and model features contributed most to the rating prediction model.

```text
outputs/feature_importance.png
```

### Interactive Network Visualization

The interactive network visualization shows player relationships based on match outcomes.

```text
outputs/interactive_network.html
```

### Node2Vec Embedding Map

The embedding map visualizes hidden player similarities learned from the competition network.

```text
outputs/node2vec_embedding_map.html
```

## Data Notes

Large raw Lichess PGN files are not included in this repository. The project expects chess match data to be placed in the `data/` folder before running the full pipeline.

The processed data files in the `data/` folder are used for graph construction, feature engineering, and model training.

## Requirements

The project dependencies are listed in `requirements.txt`.

To install them, run:

```bash
pip install -r requirements.txt
```

Common dependencies include:

* pandas
* numpy
* networkx
* node2vec
* xgboost
* scikit-learn
* matplotlib

## How to Reproduce the Project

After installing the dependencies, run:

```bash
python -m app.run_pipeline
```

The pipeline will process the available data, build the graph, engineer features, train the model, and save generated outputs.

Expected output locations:

```text
data/
models/
outputs/
```

## Skills Demonstrated

This project demonstrates experience with:

* Data cleaning and preprocessing
* Graph-based feature engineering
* Network analysis
* Machine learning regression
* Model evaluation
* Feature importance analysis
* Python project organization
* Technical documentation
* Data visualization

## Future Improvements

Potential future improvements include:

* Adding more Lichess time controls
* Expanding the dataset across more months
* Testing additional models such as Random Forest, LightGBM, or neural networks
* Adding more player behavior features
* Improving model evaluation with cross-validation
* Building a dashboard for interactive model results

## Author

Michael Chiflikyan
