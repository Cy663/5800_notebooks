# Citi Bike Station Demand Forecasting and Rebalancing Optimization
## Final Project Report

### Executive Summary

This project addresses a critical operational challenge faced by bike-sharing systems: the natural imbalance of bicycles across stations. By analyzing nearly 19 million Citi Bike trips from January through June 2024, we developed an integrated system that forecasts station-level bike demand and optimizes rebalancing operations. The solution combines time series forecasting with vehicle routing optimization to create a comprehensive decision support system for bike-sharing operators.

Our analysis identified stations experiencing the most severe imbalances, with some consistently losing or gaining significant numbers of bikes. After comparing four different time series forecasting models (ARIMA, ARIMAX, SARIMA, SARIMAX), we determined that the simpler ARIMA model produced the most accurate forecasts. We then developed and compared three different routing optimization approaches to efficiently redistribute bikes throughout the system: brute-force optimization, mixed-integer linear programming, and a heuristic-based approach using Google OR-Tools.

The complete system offers bike-sharing operators a powerful tool for predicting imbalances and planning efficient rebalancing operations across all 2,188 active Citi Bike stations, potentially reducing operational costs while improving service quality for riders.

### Introduction

#### Context

Bike-sharing systems have become an indispensable part of urban transportation infrastructure, offering convenient, eco-friendly mobility options. However, these systems face a critical operational challenge: the natural imbalance of bicycles across stations. Bikes tend to accumulate at some stations (typically in residential areas or at the bottom of hills) while others become empty (business districts, tourist attractions, or hilltops).

This imbalance creates frustration for users who arrive at empty stations or find no available docks to return their bikes. For operators, maintaining system balance requires constant "rebalancing" - physically moving bikes between stations using trucks or other vehicles - which represents a significant operational cost.

#### Research Question

This project addresses the question: How can we develop an algorithm-driven system to predict bike demand patterns and optimize rebalancing operations to maximize service quality while minimizing operational costs?

#### Rationale and Significance

Effective rebalancing is critical for maintaining service quality in bike-sharing systems, but it represents a significant operational cost. Most existing systems rely heavily on operator intuition rather than systematic data-driven approaches. By developing predictive models and optimization algorithms, we can help operators transition from reactive to proactive rebalancing, potentially:

1. Reducing operational costs through more efficient routing and resource utilization
2. Improving service availability and user satisfaction
3. Decreasing environmental impact by minimizing unnecessary vehicle movements
4. Providing data-driven insights to inform long-term system planning and expansion

#### Personal Importance

**Tianze Yin**: As a former daily bike-share rider, I've often faced the frustration of arriving at an empty station when returning from work or finding no available docks to return my bike. After a long day at work, I often had to walk to the subway station because there were no bikes left. There should be a better system to keep shared bikes accessible for everyone. Instead of relying on staff intuition, smarter software and algorithms could make rebalancing more efficient. This project directly connects to my own experience, and I'm excited to apply algorithms to improve availability for all users.

**Xu Tang**: I've always loved shared bikes—they're the cheapest and most eco-friendly way to explore a city. Feeling the wind as you ride and relying on your own energy to move makes the experience even more rewarding. However, during rush hours, the imbalance of supply and demand at stations becomes a major issue. I'm fascinated by how algorithmic approaches can optimize shared transportation systems and reduce reliance on carbon-emitting rebalancing vehicles. Previously, I've worked with machine learning algorithms for classification tasks, and this time, I have the chance to apply what I've learned by implementing ARIMA for demand forecasting. It's a great opportunity to refine my machine learning skills while tackling a real-world problem.

**Zhuoyue Lian**: Having witnessed the inefficiencies of bike-sharing systems firsthand, this project addresses a problem I've often observed: empty stations in high-demand areas while others have excess bikes. What struck me was how rebalancing decisions seemed to rely more on intuition than data analytics, creating persistent imbalances that drove users away. Seeing bikes abandoned throughout the city while certain stations remained consistently empty made me question whether there had to be a better approach. This pattern of inefficiency not only wastes valuable resources but also undermines the fundamental purpose of bike-sharing as sustainable transportation, which is why I'm eager to explore how data-driven algorithmic solutions could significantly improve operational efficiency compared to the current methods.

### Methodology

Our methodology consisted of four main components: data acquisition and cleaning, demand forecasting with time series analysis, rebalancing problem formulation, and routing optimization.

#### Data Acquisition and Cleaning

We acquired Citi Bike trip data covering January through June 2024, including over 18.9 million individual trip records across New York City. The raw dataset contained information about each trip's origin and destination stations, start and end times, bike type, and rider type.

The data cleaning process involved:
1. Merging 14 separate CSV files into a unified dataset
2. Handling missing values (51,886 rows had incomplete information)
3. Standardizing station IDs and names to fix formatting inconsistencies
4. Identifying and resolving duplicate stations with inconsistent identifiers

This resulted in a clean dataset of 18,851,994 trips with complete information across all attributes.

```python
# Sample code for data merging and cleaning
import pandas as pd
import glob
import os

# Get all Citibike CSV files
csv_files = glob.glob(os.path.join(data_dir, '2024*-citibike-tripdata*.csv'))

# Sort files chronologically
csv_files.sort()

# Merge files one by one
combined_df = None
for file in csv_files:
    df = pd.read_csv(file, low_memory=False)
    if combined_df is None:
        combined_df = df
    else:
        common_columns = list(set(combined_df.columns) & set(df.columns))
        combined_df = pd.concat([combined_df[common_columns], 
                                df[common_columns]], ignore_index=True)

# Remove rows with missing values
df_cleaned = combined_df.dropna()
```

#### Net Flow Analysis and Time Series Modeling

To understand station-level demand patterns, we calculated the "net flow" of bikes at each station for each hour:

```
Net Flow = (Bikes arriving at station) - (Bikes leaving station)
```

Positive values indicate a station gaining bikes, while negative values indicate a station losing bikes. This resulted in a time series dataset with 5,457,978 hourly flow observations across all stations.

We evaluated four different time series forecasting models:

1. **ARIMA** (Autoregressive Integrated Moving Average)
2. **ARIMAX** (ARIMA with exogenous variables)
3. **SARIMA** (Seasonal ARIMA)
4. **SARIMAX** (Seasonal ARIMA with exogenous variables)

The exogenous variables we incorporated included:
- Time-based features (hour of day, day of week)
- Weekend indicators
- Rush hour indicators (morning and evening)
- Cyclical features (sine and cosine transformations)
- Rider type distribution (member vs. casual)
- Bike type distribution (electric vs. classic)

For each model, we assessed performance across multiple forecasting horizons:
- 1-hour ahead forecasts
- 3-hour ahead forecasts
- 24-hour ahead forecasts
- 7-day ahead forecasts

Models were evaluated using multiple error metrics: Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE).

#### Rebalancing Problem Formulation

We formulated the bike rebalancing problem as a Capacitated Vehicle Routing Problem (CVRP), a variant of the Vehicle Routing Problem where vehicles have limited capacity constraints.

**Mathematical Formulation:**

Let:
- $N = \{1, 2, ..., n\}$ be the set of stations requiring rebalancing
- $V = \{0, 1, 2, ..., n\}$ be the set of all locations including the depot (0)
- $K$ be the set of available trucks
- $c_{ij}$ be the distance between locations $i$ and $j$
- $d_i$ be the demand (number of bikes needed) at station $i$
- $Q$ be the capacity of each truck

Decision variables:
- $x_{ijk} = 1$ if truck $k$ travels from location $i$ to location $j$; 0 otherwise
- $u_{ik}$ = auxiliary variable for subtour elimination

The objective is to minimize the total distance traveled:

$$\min \sum_{k \in K} \sum_{i \in V} \sum_{j \in V, j \neq i} c_{ij} x_{ijk}$$

Subject to constraints:

1. Each station must be visited exactly once:
   $$\sum_{k \in K} \sum_{i \in V, i \neq j} x_{ijk} = 1, \forall j \in N$$

2. Each truck leaves the depot at most once:
   $$\sum_{j \in N} x_{0jk} \leq 1, \forall k \in K$$

3. Flow conservation - each truck that enters a node must leave it:
   $$\sum_{i \in V, i \neq h} x_{ihk} = \sum_{j \in V, j \neq h} x_{hjk}, \forall h \in V, \forall k \in K$$

4. Capacity constraint for each truck:
   $$\sum_{i \in V} \sum_{j \in N, j \neq i} d_j x_{ijk} \leq Q, \forall k \in K$$

5. Subtour elimination constraints (Miller-Tucker-Zemlin formulation):
   $$u_{ik} - u_{jk} + |N|x_{ijk} \leq |N| - 1, \forall i,j \in N, i \neq j, \forall k \in K$$

Where:
- $u_{ik} \geq 2, \forall i \in N, \forall k \in K$
- $u_{ik} \leq |N|, \forall i \in N, \forall k \in K$

#### Routing Optimization Approaches

We implemented and compared three different approaches to solve the CVRP:

1. **Brute-Force Optimization** (for small-scale problems)
   - Generated all possible partitions of stations among trucks
   - For each valid partition, found the optimal route for each truck
   - Selected the partition with the lowest total distance
   - Limited to 10 stations due to computational complexity

2. **PuLP Mixed-Integer Linear Programming** (for medium-scale problems)
   - Implemented the mathematical formulation using PuLP and CBC solver
   - Used Miller-Tucker-Zemlin constraints for subtour elimination
   - Set a time limit of 5 minutes for solving
   - Tested on 10 stations with the highest demand

3. **Google OR-Tools Heuristic** (for large-scale problems)
   - Utilized Google's OR-Tools library for solving VRP at scale
   - Applied PATH_CHEAPEST_ARC first solution strategy
   - Employed GUIDED_LOCAL_SEARCH metaheuristic
   - Successfully scaled to all 2,188 stations in the system

Each approach was evaluated based on solution quality (total distance traveled), computational efficiency, and ability to handle problem size.

### Experiments, Results, and Analysis

#### Station Imbalance Analysis

Our initial analysis identified the stations experiencing the most significant imbalances:

| Station ID | Net Flow | Absolute Net Flow |
|------------|----------|------------------|
| 6847.02    | -4946.0  | 4946.0           |
| 6747.06    | -3057.0  | 3057.0           |
| 6955.01    | -2249.0  | 2249.0           |
| 6779.04    | 1769.0   | 1769.0           |
| 6890.06    | -1740.0  | 1740.0           |

The station with ID 6847.02 showed the largest imbalance, losing 4,946 bikes over the study period. This station became our initial focus for model development and testing.

#### Time Series Forecasting Results

After evaluating all models across different time horizons, we compiled the Mean Absolute Error (MAE) results:

| Model   | 1-hour | 3-hour | 24-hour | 7-day |
|---------|--------|--------|---------|-------|
| ARIMA   | 0.59   | 5.27   | 4.04    | 5.62  |
| ARIMAX  | 2.10   | 6.24   | 4.68    | 5.56  |
| SARIMA  | 2.14   | 5.90   | 4.29    | 5.61  |
| SARIMAX | 2.36   | 7.14   | 5.43    | 6.94  |

Surprisingly, the simpler ARIMA model outperformed the more complex models in most forecasting horizons. This suggests that the seasonal patterns and exogenous variables did not significantly improve predictive power beyond what the basic ARIMA model could capture.

For station 6847.02, our ARIMA model produced the following forecasts:

| Time Horizon | Total Net Flow | Average Per Hour |
|--------------|----------------|------------------|
| 1-hour       | -0.41          | -0.41            |
| 3-hour       | -2.52          | -0.84            |
| 24-hour      | -29.68         | -1.24            |
| 7-day        | -216.46        | -1.29            |

These forecasts indicate this station would continue to lose bikes, with a maximum depletion of 216.5 bikes over a 7-day period.

Based on these results, we selected the ARIMA model for our system-wide forecasting implementation and successfully applied it to all 2,188 active Citi Bike stations.

#### Routing Optimization Results

**1. Brute-Force Optimization (10 stations)**

Our brute-force algorithm evaluated 4,549 feasible partitions and found an optimal solution with a total distance of 38.70 km across 4 trucks. The algorithm assigned stations efficiently to each truck while respecting capacity constraints:

```
Truck 1 - Stations: [10, 5, 3], Total demand: 28/30, Distance: 11.53 km
Truck 2 - Stations: [9, 7, 1], Total demand: 28/30, Distance: 14.76 km
Truck 3 - Stations: [8, 4, 2], Total demand: 29/30, Distance: 11.38 km
Truck 4 - Stations: [6], Total demand: 9/30, Distance: 1.02 km
```

This approach provided optimal solutions but was limited to small problem instances due to its exponential computational complexity.

**2. PuLP Mixed-Integer Linear Programming (10 stations)**

The PuLP implementation found a solution with a total distance of 38.70 km, identical to the brute-force approach, confirming the optimality of both methods:

```
Truck 1: Depot -> W 45 St & 8 Ave -> W 52 St & 11 Ave -> E 68 St & Madison Ave -> Depot
  Load: 28/30 bikes, Distance: 11.53 km

Truck 2: Depot -> E 3 St & Ave A -> Depot
  Load: 9/30 bikes, Distance: 1.02 km

Truck 3: Depot -> Old Slip & South St -> Greenwich Ave & Charles St -> 8 Ave & W 31 St -> Depot
  Load: 29/30 bikes, Distance: 11.38 km

Truck 4: Depot -> Stagg St & Union Ave -> Grand Army Plaza & Plaza St West -> Schermerhorn St & Court St -> Depot
  Load: 28/30 bikes, Distance: 14.76 km
```

While providing optimal solutions, this approach still faced computational challenges with larger instances, taking approximately 2 minutes to solve the 10-station problem.

**3. Google OR-Tools Heuristic (All 2,188 stations)**

The OR-Tools implementation successfully scaled to the full problem, handling all 2,188 stations with negative bike flow (expected outflow). With a fleet of 66 trucks (capacity 30 bikes each), it generated routes with a total distance of 1,247.56 km in just 0.17 seconds:

```
Solution summary:
Total distance: 1247.56 km
Total bikes delivered: 1976
Trucks used: 66 out of 66
Maximum bikes per truck: 30 out of capacity 30
Average bikes per truck: 29.94
Average distance per truck: 18.90 km
```

Sample truck routes demonstrated efficient planning, with each truck serving a cluster of nearby stations:

```
Truck 2: 21 stations, 30 bikes, 36.39 km
  Route: Depot -> Park Ave & E Tremont Ave (3 bikes) -> Hughes Ave & Oak Tree Pl (1 bikes) -> ... -> W 238 St & Tibbett Ave (1 bikes) -> Depot
```

This approach proved to be the most scalable and practical for real-world implementation across the entire Citi Bike system.

#### Comparative Analysis

Our experiments revealed important trade-offs between solution quality and computational efficiency:

1. **Brute-Force Optimization**:
   - Pros: Guaranteed optimal solutions
   - Cons: Exponential complexity, limited to ~10 stations
   - Best use case: Small-scale pilot testing or critical high-priority stations

2. **Mixed-Integer Linear Programming**:
   - Pros: Optimal or near-optimal solutions with mathematical guarantees
   - Cons: Computational complexity still limits scale to dozens of stations
   - Best use case: Medium-scale problems or when optimality guarantees are required

3. **OR-Tools Heuristic**:
   - Pros: Extremely scalable, fast solutions for thousands of stations
   - Cons: No optimality guarantee, solutions may be suboptimal
   - Best use case: System-wide daily operational planning

The combination of ARIMA forecasting with OR-Tools routing optimization provided the most practical approach for real-world implementation across the entire Citi Bike system.

### Conclusion and Future Work

#### Key Findings

1. **Predictable Station Imbalances**: Our analysis confirmed that bike stations exhibit consistent patterns of imbalance, with some stations persistently losing or gaining bikes over time. These patterns are predictable using time series forecasting techniques.

2. **Model Simplicity Advantage**: Contrary to our initial expectations, the simpler ARIMA model outperformed more complex alternatives (ARIMAX, SARIMA, SARIMAX) for demand forecasting. This suggests that adding seasonal components and exogenous variables did not significantly improve predictive power for this particular application.

3. **Routing Algorithm Trade-offs**: We demonstrated a clear trade-off between solution quality and computational tractability in routing optimization. While exact methods (brute-force and MILP) guarantee optimal solutions for small instances, heuristic approaches like OR-Tools offer practical solutions for system-wide optimization.

4. **Integrated System Viability**: By combining time series forecasting with routing optimization, we successfully created an end-to-end system capable of predicting station-level demands and optimizing rebalancing operations across all 2,188 active Citi Bike stations.

#### Limitations

1. **Weather Impact**: Our current forecasting models do not incorporate weather data, which can significantly impact bike usage patterns. Incorporating precipitation, temperature, and other weather variables could improve prediction accuracy.

2. **Traffic Conditions**: The routing optimization does not account for real-time traffic conditions, which could affect actual travel times between stations. A more sophisticated approach might incorporate time-dependent travel times.

3. **Operational Constraints**: Our model simplifies some operational aspects, such as loading/unloading times, staffing constraints, and work shifts. A more comprehensive model would include these factors.

4. **Fleet Heterogeneity**: We assumed a homogeneous fleet of trucks with identical capacities, whereas real fleets might include vehicles with different capacities and operational costs.

#### Future Work

1. **Weather Integration**: Incorporate weather forecasts as exogenous variables in the demand prediction models, potentially using API data from weather services.

2. **Dynamic Rebalancing**: Develop algorithms for real-time adjustments to rebalancing plans as new data becomes available during the day, allowing for more responsive operations.

3. **Multi-Objective Optimization**: Extend the routing optimization to consider multiple objectives beyond just distance minimization, such as service level guarantees, environmental impact, or operational costs.

4. **User Incentives**: Explore complementary approaches to physical rebalancing, such as user incentive systems that encourage rides in directions that naturally rebalance the system.

5. **System Design Recommendations**: Use the accumulated data and insights to provide recommendations for optimal station sizing and placement in future system expansions.

#### Individual Learning Reflections

**Tianze Yin**: This project significantly enhanced my understanding of combinatorial optimization problems and their real-world applications. Implementing three different approaches to solve the CVRP—from brute-force to advanced heuristics—taught me valuable lessons about algorithm design and trade-offs between optimality and computational efficiency. The most challenging aspect was scaling our solution to handle thousands of stations, which required me to explore specialized libraries like Google OR-Tools. This knowledge will be directly applicable to my future work in transportation planning and logistics optimization, where similar routing problems are common. The experience of translating a mathematical formulation into working code that can solve real-world problems has been incredibly rewarding.

**Xu Tang**: Working on time series forecasting for this project has been an enlightening journey. I've deepened my understanding of ARIMA models and their variations, learning how to properly evaluate and compare different forecasting approaches. What surprised me most was that sometimes simpler models perform better than complex ones—a valuable lesson in avoiding unnecessary complexity. The skills I've gained in time series analysis will be directly applicable to my planned career in data science, where forecasting is a fundamental technique across multiple domains. This project also improved my data preprocessing abilities, as handling and cleaning the massive Citi Bike dataset required careful attention to detail and efficient coding practices.

**Zhuoyue Lian**: Through this project, I gained invaluable experience in handling large real-world datasets and the challenges they present. The data acquisition and cleaning phase taught me practical skills in dealing with inconsistencies, missing values, and merging multiple data sources. I learned that data preparation often constitutes the most time-consuming yet critical part of any analytical project. The visualization techniques I developed will be directly applicable to my future work in data analytics, where communicating complex patterns effectively is essential. This project reinforced my belief in the power of data-driven decision-making and inspired me to further explore how analytics can improve urban transportation systems.

### Code Implementation

Our codebase consists of three main components, each handling a distinct aspect of the bike rebalancing system:

1. **Data Processing Module**: Responsible for cleaning, transforming, and preparing the raw trip data for analysis.

2. **Forecasting Module**: Implements the time series models (ARIMA, ARIMAX, SARIMA, SARIMAX) and evaluates their performance.

3. **Routing Optimization Module**: Contains the three different approaches for solving the CVRP (brute-force, MILP, and OR-Tools).

The code follows a modular design with clear separation of concerns, well-documented functions, and consistent naming conventions. Each component can be run independently or as part of the integrated system.

Example of our routing optimization code implementation:

```python
def solve_cvrp_ortools(stations, depot_location, num_trucks, truck_capacity):
    """
    Solve the Capacitated Vehicle Routing Problem using Google OR-Tools.
    
    Args:
        stations (list): List of dictionaries containing station information
                        (name, demand, lat, lng)
        depot_location (tuple): Latitude and longitude of the depot
        num_trucks (int): Number of available trucks
        truck_capacity (int): Capacity of each truck in number of bikes
        
    Returns:
        dict: Solution details including routes, distances, and load information
    """
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(stations) + 1, num_trucks, 0)
    
    # Create Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    # Define distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        
        # Get coordinates
        if from_node == 0:  # Depot
            from_lat, from_lng = depot_location
        else:
            from_lat = stations[from_node - 1]["lat"]
            from_lng = stations[from_node - 1]["lng"]
            
        if to_node == 0:  # Depot
            to_lat, to_lng = depot_location
        else:
            to_lat = stations[to_node - 1]["lat"]
            to_lng = stations[to_node - 1]["lng"]
        
        # Calculate haversine distance
        return int(haversine_distance(from_lat, from_lng, to_lat, to_lng))
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add Capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if from_node == 0:  # Depot has no demand
            return 0
        return stations[from_node - 1]["demand"]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [truck_capacity] * num_trucks,  # vehicle capacity
        True,  # start cumul to zero
        'Capacity')
    
    # Set solution parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 300  # 5 minutes
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    # Extract and return solution
    if solution:
        return extract_solution(routing, manager, solution, stations, depot_location)
    else:
        return {"status": "No solution found"}
```

This modular approach allows for easy maintenance, testing, and extension of the codebase as the project evolves.
