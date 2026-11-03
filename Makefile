.PHONY: install run ui feast-ui lint smoke clean

install:            ## install dependencies into the active environment
	pip install -r requirements.txt

run:                ## full pipeline on data/raw/athletes.csv.zip
	python -m src.pipeline

ui:                 ## MLflow UI (port 5050: macOS AirPlay already uses 5000)
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050

feast-ui:           ## Feast UI for the registered features
	cd feature_repo && feast ui

lint:
	ruff check .

smoke:              ## end-to-end test on synthetic data (no course file needed)
	python tests/make_synthetic_athletes.py data/raw/athletes_synthetic.csv
	python -m src.pipeline --config tests/config_smoke.yaml

clean:              ## remove local MLflow / Feast / CodeCarbon state and processed data
	rm -rf mlflow.db mlruns mlartifacts emissions feature_repo/data data/processed/*.parquet
