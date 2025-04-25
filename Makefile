gd:
	poetry run python -B generate_data_set.py
up: 
	poetry run python -B train.py

code:
	poetry run nvim
