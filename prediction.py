import pandas as pd
import matplotlib.pyplot as plt
import json
from utils import load_dataset, regression_line, load_model, ask_mileage

DATASET_PATH = 'Données.csv'


def plot_model(data, x_line, y_line):

	plt.scatter(data["km"], data["price"], label='Data')
	plt.plot(x_line, y_line, color='red', label='Model')
	
	plt.xlabel('Mileage')
	plt.ylabel('Price')
	plt.legend()
	
	plt.show()


def prediction(mileage, model):

	theta_0 = model["theta_0"]
	theta_1 = model["theta_1"]
	
	km_mean = model["km_mean"]
	km_std = model["km_std"]
	
	price_mean = model["price_mean"]
	price_std = model["price_std"]

	mileage = (mileage - km_mean) / km_std
	
	y_norm = theta_0 + (theta_1 * mileage)
	predicted_price = y_norm * price_std + price_mean

	return predicted_price


def main():
	
	try:
		model = load_model()
		data = load_dataset()
		mileage = ask_mileage()
		predicted_price = prediction(mileage, model)
		print(f"Predicted Price : {predicted_price}")
		x_line, y_line = regression_line(model, data)

		plot_model(data, x_line, y_line)

	except Exception as e:
		print(e)

if __name__ == '__main__':
	main()	