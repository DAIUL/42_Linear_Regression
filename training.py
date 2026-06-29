import json
from utils import load_dataset


# Modele ---> h(x) = tetha0 * x0 + tetha1 * x1
# Avec toujours x0 = 1 pour la normalisation, et x1 = mileage

# Chaque boucle tetha0 := tetha0 - alpha * 1/m * sigma(h(x_i) - y_i)
# Chaque boucle tetha1 := tetha1 - alpha * 1/m * sigma((h(x_i) - y_i) * x1_i)
# Avec ici m = nb d'element et alpha = learning rate (decide arbitrairement ni trop haut pour pas exploser, ni trop bas pour avoir un pas significatif a chaque iteration)
# Donc calcul de la soustraction du produit de la moyenne des erreurs et du learning rate a tetha0 pour tetha0
# et calcul de la soustraction du produit de la moyenne des erreurs * valeur de la feature de l'exemple et du learning rate a tetha1 pour tetha1

# Ici on utilise le 'batch descent gradiant' donc on change les valeurs des tetha apres avoir vu tout le dataset a chaque iteration

# loss_function = h(x_i) - y_i (* x_i pour tetha1)
# cost_function = 1/m * loss


def training(
	dataset,
	learning_rate = 0.01,
	tolerance = 1e-5,
	max_epoch = 10_000
):
	
	theta_0 = 0
	theta_1 = 0
	epoch = 0
	
	prev_cost = float("inf")
	curr_cost = 0

	km_mean = dataset["km"].mean()
	km_std = dataset["km"].std()

	price_mean = dataset["price"].mean()
	price_std = dataset["price"].std()
	
	m = len(dataset)

	while abs(prev_cost - curr_cost) > tolerance and epoch != max_epoch:

		sum_errors_0 = 0
		sum_errors_1 = 0
		cost = 0

		for i in range(m):

			km = (dataset["km"].iloc[i] - km_mean) / km_std
			price = (dataset["price"].iloc[i] - price_mean) / price_std

			prediction = theta_0 + (theta_1 * km)
			error = prediction - price
			cost += error ** 2
			sum_errors_0 += error
			sum_errors_1 += error * km

		theta_0 = theta_0 - learning_rate * (sum_errors_0/m)
		theta_1 = theta_1 - learning_rate * (sum_errors_1/m)
		
		prev_cost = curr_cost
		curr_cost = cost / (2 * m)
		
		epoch += 1

	model = {
		"theta_0": theta_0,
		"theta_1": theta_1,
		"km_mean": km_mean,
		"km_std": km_std,
		"price_mean": price_mean,
		"price_std": price_std,
		"curr_cost": curr_cost,
		"epoch": epoch
	}

	with open("model.json", "w") as f:
		json.dump(model, f)
	
	return

def main():
	try:
		dataset = load_dataset()

		training(dataset)

		print('Training done <3')
	except Exception as e:
		print(e)

if __name__ == '__main__':	
	main()