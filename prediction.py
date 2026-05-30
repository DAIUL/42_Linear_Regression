import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

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
	
	with open('model.json', 'r') as f:
		model = json.load(f)

	data = pd.read_csv("Données.csv")

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
	
	mileage = float(input("Mileage : "))

	predicted_price = prediction(mileage, model)

	print(f"Predicted Price : {predicted_price}")

	plt.scatter(data["km"], data["price"], label='Data')
	plt.plot(data["km"], pred_y, color='red', label='Model')
	plt.xlabel('Mileage')
	plt.ylabel('Price')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()	