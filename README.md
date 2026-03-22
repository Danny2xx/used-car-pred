# Used Car Price Estimator

This project predicts used-car resale prices from raw listing data and packages the result as both:

- a portfolio notebook: `Used_Car_Price_Portfolio_Pipeline.ipynb`
- a Streamlit frontend: `app.py`

The trained artifact already lives in `models/used_car_price_champion.joblib`, and the app uses the same raw-input contract that the notebook trained on.

## What is in the project

- `Used_Car_Price_Portfolio_Pipeline.ipynb`: end-to-end regression workflow
- `app.py`: polished Streamlit frontend for inference and model storytelling
- `vehicle_features.py`: reusable feature-engineering logic and unpickling support for the saved model
- `models/used_car_price_champion.joblib`: champion model artifact
- `reports/`: benchmark, tuning, and final evaluation outputs
- `artifacts/`: input contract and sample inference payloads

## Modeling summary

- Problem: predict `selling_price` from raw car-listing fields
- Data source: `Car details v3.csv`
- Pipeline highlights:
  - deduplication before splitting
  - targeted EDA
  - parsing of `mileage`, `engine`, `max_power`, and `torque`
  - engineered vehicle-age and usage-intensity features
  - hybrid text-plus-tabular experimentation
  - multiple model comparison and tuning

Champion model:

- `HistGradientBoostingRegressor` wrapped in a deployment-friendly preprocessing pipeline

Held-out test performance:

- `R2`: `0.897`
- `MAE`: `73,453`
- `RMSE`: `164,725`

## Run the app locally

1. Create and activate your environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the frontend:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and connect your GitHub repository.
4. Set the entrypoint file to `app.py`.
5. In **Advanced settings**, choose **Python 3.11**.
6. Deploy.

Notes:

- Community Cloud installs dependencies from `requirements.txt`, which should live in the repo root or next to your entrypoint file.
- Streamlit's docs note that the Python version is selected in the deployment dialog, not through `runtime.txt`.
- If you ever need to change the Python version later, Streamlit's docs say you must delete and redeploy the app.

## What the frontend does

- accepts raw listing-style inputs
- loads the saved model artifact
- predicts a resale price
- shows a typical error band using held-out MAE
- displays parsed feature values from the raw input
- includes a simple what-if scenario explorer
- shows benchmark context from the notebook outputs

## Notes

- The app displays prices in the dataset's original scale, which corresponds to Indian rupees.
- Because the saved model contains a custom transformer, `vehicle_features.py` registers the class needed to load the joblib artifact safely outside the notebook.
- If you retrain the notebook and resave the model, keep the same raw input columns so the Streamlit form stays aligned.
- The cloud deployment only needs the app dependencies, so plotting libraries used in the notebook are intentionally left out of `requirements.txt`.
