up: 
	poetry run python -B my_own_easy_llm.py
gd:
	poetry run python -B generate_data_set.py
best: 
	poetry run python -B train.py

code:
	poetry run nvim
