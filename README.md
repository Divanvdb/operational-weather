# Implementation of Neural LAM by Divan van der Bank - Creating SA-GNN

## Wind Speed Forecasting with Limited-Area AI Model

This project develops a **Graph Neural Network (GNN)-based wind speed forecasting model** using a limited-area **Artificial Intelligence Weather Prediction (AIWP)** approach. Designed to tackle wind power variability challenges, the model will leverage AIWP advancements to match the skill of traditional **Numerical Weather Prediction (NWP)** and leading AIWP models, while requiring *significantly less computational power*. The model will be validated on a custom benchmarking platform, aiming to achieve performance comparable to state-of-the-art methods for wind speed forecasting in South Africa (2020 data). A case study will demonstrate its operational potential at a wind farm, providing accurate, cost-effective forecasts to support renewable energy deployment and contribute to a sustainable energy future.

This model, called **SA-GNN**, is based on work done by *Oskarsson, Joel and Landelius, Tomas and Lindsten, Fredrik* and details the modification and validation of their **neural-lam model** on `ERA5 data` for the `region of South Africa`. The results of this model will be compared to **current leading AI weather prediction models** and numeric weather prediction models using a modified version of **WeatherBench** by *Rasp, Stephan*.

Find the original **NeuralLAM** repository at [NeuralLAM](https://github.com/mllam/neural-lam/tree/main) and the paper for the model on [arxiv](https://arxiv.org/abs/2309.17370)

Find the original **WeahterBench** repository at [WeatherBench](https://github.com/google-research/weatherbench2)

## Dataset description and goal

### Overview
The dataset consists of historical wind speed data for South Africa, sourced from **ERA5** reanalysis. It spans the period from **2010 to 2020** and is used for machine learning-based (Graph Neural Network) spatial wind speed forecasting. The **first nine years (2010–2019)** are used for **training, validation, and testing**, while the **year 2020** serves as a **holdout set** for comparison against the most recent global models.

### Spatial and Temporal Resolution
- Latitude × Longitude: 49 × 69 grid points over South Africa
- Time: 6-hourly intervals from 2010 to 2020

### Features and Targets
1. Input features:
    - 850 hPa Atmospheric Variables:
    - z850.0hPa – Geopotential height
    - r850.0hPa – Relative humidity
    - t850.0hPa – Temperature
    - u850.0hPa – Zonal wind component
    - v850.0hPa – Meridional wind component
    - w850.0hPa – Vertical wind velocity
    - wind_speed850.0hPa – Wind speed at 850 hPa
2. Forcings:
    - Hour of the day (0-23)
    - Month of the year (1-12)
    - Land-Sea Mask (binary indicator for land and ocean regions)
3. Output Target:
    - Outputs the **next weather state** and autoregressively produces forecasts for a specified lead time

### **Xarray Dataset Description**  

#### **Dimensions**  
- **time:** `14,608` timestamps (from `2010-01-01` to `2019-12-31 18:00 UTC` at `6-hourly intervals`)  
- **pressure_level:** `1` (850 hPa)  
- **y:** `49` latitudinal grid points (ranging from `-35.0° to -23.0°`)  
- **x:** `69` longitudinal grid points (ranging from `16.0° to 33.0°`)  

#### **Coordinates**  
- **time:** `datetime64[ns]` (timestamps for each time step)  
- **pressure_level:** `float64` (single level: `850 hPa`)  
- **y:** `float64` (latitude values from `-35.0° to -23.0°`)  
- **x:** `float64` (longitude values from `16.0° to 33.0°`)  

#### **Data Variables**  
Each variable has dimensions `(time, pressure_level, y, x)` unless otherwise specified.  

- **z:** `float32` – Geopotential height at `850 hPa`  
- **r:** `float32` – Relative humidity at `850 hPa`  
- **t:** `float32` – Temperature at `850 hPa`  
- **u:** `float32` – Zonal wind component at `850 hPa`  
- **v:** `float32` – Meridional wind component at `850 hPa`  
- **w:** `float32` – Vertical wind velocity at `850 hPa`  
- **wind_speed:** `float32` – Wind speed at `850 hPa` (computed from `u` and `v`)  

### **Additional Forcing Variables (2D Grid-Based)**
- **hour:** `(y, x, time) int64` – Hour of the day (`0-23`)  
- **month:** `(y, x, time) int64` – Month of the year (`1-12`)  
- **lsm (Land-Sea Mask):** `(y, x) float32` – Binary indicator (`1` for land, `0` for sea)  

### **Key Insights**
- The dataset is **4D** (`time, pressure level, latitude, longitude`), with all variables stored at a **single pressure level (850 hPa)**.  
- The **land-sea mask (lsm)** is static across time and stored as a `(y, x)` 2D field.  
- **Hour and month** are stored as `(y, x, time)`, suggesting they may vary spatially (e.g., adjusted for local time zones).  
- The dataset is structured efficiently for **spatial-temporal forecasting**, making it ideal for machine learning applications.  

## Graph Neural Network

# Wind Speed Forecasting with Limited-Area AI Model

This project develops a **Graph Neural Network (GNN)-based wind speed forecasting model** using a limited-area Artificial Intelligence Weather Prediction (AIWP) approach. Designed to tackle wind power variability challenges, the model will leverage AIWP advancements to match the skill of traditional Numerical Weather Prediction (NWP) and leading AIWP models, while requiring significantly less computational power. The model will be validated against a custom benchmark, aiming to achieve performance comparable to state-of-the-art methods for wind speed forecasting in South Africa (2020 data). A case study will demonstrate its operational potential at a wind farm, providing accurate, cost-effective forecasts to support renewable energy deployment and contribute to a sustainable energy future.

## Model Architecture

The model follows an **Encoder-Processor-Decoder** framework, adapted from the GraphCast model, with enhancements for limited-area forecasting:

- **Graph Construction**: Constructs a multi-scale graph structure using regular quadrilateral meshes tailored to the limited area (e.g., SA region) which can be seen in the figure below:

![alt text](images/multiscale_graph.png)

- **Encoder**: Employs Multi-Layer Perceptrons (MLPs) with one hidden layer (Swish activation and LayerNorm) to encode input weather states from grid nodes to mesh nodes, incorporating static features and boundary forcing.
- **Processor**: Utilizes GNN **message-passing layers** (Interaction Networks) to process information across mesh nodes. 
- **Decoder**: Uses MLPs to decode processed mesh features back to grid nodes, producing one-step predictions (e.g., 3h intervals) that can be unrolled autoregressively for multi-step forecasts (e.g., 36h).

This architecture will combine GNN efficiency with LAM flexibility, targeting high-resolution forecasts with lower energy demands than traditional NWP.

## Goals and Validation

The model will aim to:
- Match or approximate the forecasting skill of operational NWP systems (e.g., MEPS or ERA5 or IFS HRES) for wind speed.

Validation will involve training on historical NWP data (e.g., ERA 2010-2020) and evaluating performance via RMSE and qualitative comparisons, such as in the figure below, over lead times up to 36h, with a focus on a limited area like the SA region with wind power forecasting in mind.

![alt text](images/metric_graphs.png)

## Training Computation

## Progress towards completing the model:

Currently in the training phase of the development

![alt text](images/progress.png)


## And the first outputs of the model

[![Weather Model Demo](images/output_1.png)](images/forecast_1.mp4)
