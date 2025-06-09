

# 🎵 Transformer-Based Music Popularity Prediction

This repository contains a PyTorch-based implementation of a **Transformer model** to predict the **popularity of Spotify songs** based on audio features.

## 🚀 Overview

The model learns to predict the `track_popularity` score (0–100) using a set of numerical audio features extracted from the `spotify_songs.csv` dataset. It uses a **Transformer encoder** architecture — originally developed for NLP tasks — and adapts it for regression.

## 🧠 Model Architecture

* Transformer Encoder with:

  * Positional projection from 1D to `d_model` (default: 32)
  * Multi-head self-attention (`nhead` = 4)
  * Two encoder layers (`num_layers` = 2)
* Output layer flattens sequence and predicts popularity as a single scalar value.

## 📊 Features Used

The following features are used as input:

* `danceability`
* `energy`
* `loudness`
* `speechiness`
* `acousticness`
* `instrumentalness`
* `liveness`
* `valence`
* `tempo`

Target:

* `track_popularity`

## 🧪 Dataset

This project expects a CSV file named `spotify_songs.csv` with the columns listed above.
Missing values are dropped during preprocessing.

> **Note:** Update the CSV path in the script if needed.

## 🛠 Requirements

* Python 3.8+
* PyTorch
* scikit-learn
* pandas
* matplotlib

You can install dependencies using:

```bash
pip install torch scikit-learn pandas matplotlib
```

## 🧰 How It Works

### 1. Data Preprocessing

* Features are standardized using `StandardScaler`.
* Target is kept as raw values (0–100).
* Data is split into training and testing sets.
* Features are reshaped into sequence format for the Transformer.

### 2. Training

* The model is trained using **Mean Squared Error (MSE)** loss.
* Optimizer: Adam
* Number of epochs: 15

### 3. Evaluation

* After training, the model is evaluated on the test set.
* Final MSE loss is printed.
* A training loss chart is displayed.

## 📈 Results

A plot of training loss over epochs helps visualize the learning curve.

<p align="center">
  <img src="example_loss_plot.png" alt="Loss Plot" width="500"/>
</p>

> You can save the loss plot in your script using `plt.savefig("loss_plot.png")`.

## 📂 File Structure

```
├── spotify_songs.csv       # Dataset file
├── transformer_regression.py  # Main training script
└── README.md               # This file
```

## 📌 Example Usage

To run the script:

```bash
python transformer_regression.py




