# Dataset Description
## Overview
The dataset consists of historical wind speed data for South Africa, sourced from **ERA5** reanalysis. It spans the period from **2010 to 2020** and is used for machine learning-based (Graph Neural Network) spatial wind speed forecasting. The **first nine years (2010–2019)** are used for **training, validation, and testing**, while the **year 2020** serves as a **holdout set** for comparison against the most recent global models.

## Spatial and Temporal Resolution
- Latitude × Longitude: 49 × 69 grid points over South Africa
- Time: 6-hourly intervals from 2010 to 2020

## Features and Targets
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
    - Wind speed at future time steps for `x` number of timesteps

# **Xarray Dataset Description**  

## **Dimensions**  
- **time:** `14,608` timestamps (from `2010-01-01` to `2019-12-31 18:00 UTC` at `6-hourly intervals`)  
- **pressure_level:** `1` (850 hPa)  
- **y:** `49` latitudinal grid points (ranging from `-35.0° to -23.0°`)  
- **x:** `69` longitudinal grid points (ranging from `16.0° to 33.0°`)  

## **Coordinates**  
- **time:** `datetime64[ns]` (timestamps for each time step)  
- **pressure_level:** `float64` (single level: `850 hPa`)  
- **y:** `float64` (latitude values from `-35.0° to -23.0°`)  
- **x:** `float64` (longitude values from `16.0° to 33.0°`)  

## **Data Variables**  
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

## **Key Insights**
- The dataset is **4D** (`time, pressure level, latitude, longitude`), with all variables stored at a **single pressure level (850 hPa)**.  
- The **land-sea mask (lsm)** is static across time and stored as a `(y, x)` 2D field.  
- **Hour and month** are stored as `(y, x, time)`, suggesting they may vary spatially (e.g., adjusted for local time zones).  
- The dataset is structured efficiently for **spatial-temporal forecasting**, making it ideal for machine learning applications.  

