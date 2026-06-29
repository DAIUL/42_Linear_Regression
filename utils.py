import pandas as pd
import json
from pathlib import Path

MODEL_PATH = 'model.json'
DATASET_PATH = 'Données.csv'


def load_model() -> dict:

	if not Path(MODEL_PATH).exists():
		return {
			"theta_0": 0,
			"theta_1": 0,
			"km_mean": 0,
			"km_std": 1,
			"price_mean": 0,
			"price_std": 1,
			"curr_cost": 0,
			"epoch": 0
		}

	with open(MODEL_PATH, 'r') as f:
		model = json.load(f)

	required = {
		"theta_0",
		"theta_1",
		"km_mean",
		"km_std",
		"price_mean",
		"price_std",
		"curr_cost",
		"epoch"
	}

	missing = required - model.keys()

	if missing:
		raise ValueError(
			f"Missing keys: {missing}"
		)
	
	return model


def load_dataset() -> pd.DataFrame:

	if not Path(DATASET_PATH).exists():
		raise FileNotFoundError(
			f"{DATASET_PATH} does not exists"
		)

	data = pd.read_csv(DATASET_PATH)

	if data.empty:
		raise ValueError(
			'Dataset empty'
		)
	
	return data


def regression_line(model: dict, data: pd.DataFrame) -> list:
	
	theta_0 = model["theta_0"]
	theta_1 = model["theta_1"]

	km_mean = model["km_mean"]
	km_std = model["km_std"]	

	price_mean = model["price_mean"]
	price_std = model["price_std"]

	km_vals = data["km"].values
	km_norm = (km_vals - km_mean) / km_std
	pred_y_norm = theta_0 + (theta_1 * km_norm)
	pred_y = pred_y_norm * price_std + price_mean

	return km_vals, pred_y


def ask_mileage() -> float:

	while True:
		try:
			return float(input('Mileage : '))
		except ValueError:
			print('Please enter a valid number')