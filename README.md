# Car Prices Prediction

This project trains a machine learning model to predict used car prices from a tabular dataset and exposes a Streamlit app for interactive inference.

## Project structure

- `app.py` - Streamlit app for making predictions and visualizing results.
- `train.py` - Training script that prepares data, trains the model, and saves artifacts.
- `used_cars.csv` - Dataset of used cars used for training and evaluation.
- `data_visulization.ipynb` - Notebook with exploratory data analysis and visualization.

## Requirements

Typical Python packages used:

- pandas
- numpy
- scikit-learn
- joblib
- streamlit

Install with:

```bash
python -m venv .venv
# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -U pip
pip install pandas numpy scikit-learn matplotlib seaborn joblib streamlit
```

## Usage

Train the model:

```bash
python train.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Dataset

The dataset `used_cars.csv` should be placed in the project root. The training script expects this CSV and will read it directly.

## Notes

- Edit `train.py` to change model type, features, or hyperparameters.
- Consider adding a `requirements.txt` by running `pip freeze > requirements.txt` after installing packages.


