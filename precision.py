import pandas as pd
import json
from prediction import prediction


def main():

	with open('model.json', 'r') as f:
		model = json.load(f)
	
	data = pd.read_csv("Données.csv")

	m = len(data)

	ss_res = 0
	ss_tot = 0
	sum_model_error = 0

	price_mean = model['price_mean']

	for i in range(m):
		
		pred = prediction(data["km"].iloc[i], model)

		ss_tot += (data["price"].iloc[i] - price_mean) ** 2
		ss_res += (data["price"].iloc[i] - pred) ** 2
		
		sum_model_error += abs(pred - data["price"].iloc[i])

	r2 = 1 - (ss_res / ss_tot)
	mae = sum_model_error / m

	print(f"MAE : {mae}")
	print(f"R2 : {r2}")


if __name__ == '__main__':
	main()