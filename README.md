
# Time Series Forecasting with CNN and LSTM for Melbourne's Minimum Temperatures

## Overview

This project focuses on forecasting Melbourne's daily minimum temperatures using deep learning models, specifically Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The dataset comprises daily minimum temperatures recorded in Melbourne from 1981 to 1990.

## Dataset

The dataset used is the [Daily Minimum Temperatures in Melbourne](https://www.kaggle.com/datasets/ekaratnida/daily-minimum-temperatures-in-melbourne) dataset, which includes:

- **Time Period:** 1981 to 1990
- **Frequency:** Daily
- **Feature:** Minimum temperature of the day in degrees Celsius

## Models Implemented

1. **Convolutional Neural Network (CNN):**
   - Extracts spatial features from the time series data.
   - Captures local patterns and trends.

2. **Long Short-Term Memory (LSTM) Network:**
   - Captures temporal dependencies in the data.
   - Effective for sequential data and time series forecasting.

3. **Hybrid CNN-LSTM Model:**
   - Combines CNN and LSTM to leverage both spatial and temporal features.
   - Aims to improve forecasting accuracy by capturing complex patterns.

## Repository Structure

- `data/`
  - Contains the dataset file `daily-min-temperatures.csv`.
- `notebooks/`
  - Jupyter notebooks with data exploration, preprocessing, and model implementation.
- `models/`
  - Saved trained models for future inference.
- `results/`
  - Plots and metrics evaluating model performance.
- `README.md`
  - Project overview and instructions.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Install the required packages using:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

### Running the Project

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ahmad786writes/Time-Series-Forecasting-with-CNN-and-LSTM-for-Melbourne-s-Minimum-Temperatures.git
   cd Time-Series-Forecasting-with-CNN-and-LSTM-for-Melbourne-s-Minimum-Temperatures
   ```

2. **Navigate to the `notebooks/` directory and open the Jupyter notebooks:**

   ```bash
   jupyter notebook
   ```

3. **Follow the notebooks to:**
   - Load and preprocess the data.
   - Build and train the models.
   - Evaluate model performance.

## Results
![results](results.png)

The models were evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). The hybrid CNN-LSTM model outperformed individual CNN and LSTM models, demonstrating the effectiveness of combining spatial and temporal feature extraction for time series forecasting.

## References

- [Daily Minimum Temperatures in Melbourne Dataset](https://www.kaggle.com/datasets/ekaratnida/daily-minimum-temperatures-in-melbourne)
- [Deep Learning for Time Series Forecasting: A Review](https://ieeexplore.ieee.org/document/8035685)

## Acknowledgments

Special thanks to the contributors of the dataset and the open-source community for providing tools and resources that facilitated this project.
