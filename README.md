# 5800_notebooks

README for our notebook

Project Structure
This project is organized into 3 main sections, each represented by a folder:
Our codebase is organized into three main sections, each responsible for a core component of the bike rebalancing system:

1. **Data Processing**  
   Handles cleaning, transforming, and preparing raw Citi Bike trip data.  
   - **data_cleaning_v4.ipynb** – Cleans and preprocesses raw trip records for further analysis.

2. **Forecasting**  
   Implements and evaluates time series models (ARIMA, ARIMAX, SARIMA, SARIMAX).  
   - **Arima_Xu.ipynb** – Further exploration for dataset for Arima optimization approaches
   - **ArimaWithDifferentIntervals.ipynb** – Implementation of different parameters for Arima optimization
   - **FeatureEngineeringFinal.ipynb** – Creates additional exogenous features for ARIMAX and SARIMAX models.  
   - **Demand Forecast with ARIMA.ipynb** – Compares models, performs evaluations, and saves forecast results to `citibike_station_demand_forecasts.csv`.

3. **Routing Optimization**  
   Applies three different CVRP solutions: brute-force, MILP (PuLP), and Google OR-Tools.  
   - **Routing Optimization.ipynb** – Contains all optimization implementation, routing results and details.
