# Patient Flow Optimization at Urban Multi-Specialty Clinic

This project implements an AI-driven solution to optimize patient flow at an urban multi-specialty clinic, specifically addressing peak-hour congestion. The solution utilizes machine learning to predict patient delays, dynamically adjust appointment schedules, and improve patient communication.

## Problem Statement

The Jayanagar Specialty Clinic in Bangalore experiences significant patient congestion during peak evening hours (5-8 PM). This results in long wait times, impacting patient satisfaction and clinic efficiency.

## Solution Overview

This project addresses the problem by implementing the following key components:

* **Predictive Load Balancing:**
    * Uses a `RandomForestRegressor` model to predict patient delays based on doctor schedules, consultation durations, and appointment times.
    * Dynamically adjusts appointment schedules to distribute patient load evenly across doctors, minimizing wait times.
* **Time-Slot Optimization:**
    * Optimizes appointment time slots based on doctor consultation patterns and predicted delays.
    * Accounts for variations in consultation durations and doctor-specific delays to prevent scheduling conflicts.
* **Patient Communication:**
    * Simulates SMS notifications to provide patients with realistic estimated wait times.
    * Improves patient experience by increasing transparency and managing expectations.
* **Early Arrival Handling:**
    * Implements a queueing system that prioritizes patients that arrive early, without disrupting the schedule of other patients.
* **Patient Arrival Forecasting:**
    * Implements a basic LSTM model to forecast patient arrival numbers.

## Implementation Details

* **Programming Language:** Python
* **Machine Learning Libraries:** scikit-learn, TensorFlow/Keras
* **Data Simulation:** Simulated patient appointment data is used for demonstration purposes.
* **Model Training:**
    * The `RandomForestRegressor` model is trained to predict patient delays.
    * The `LSTM` model is trained to forecast patient arrival numbers.
* **Dynamic Scheduling:** The `optimize_schedule` function adjusts appointment times based on predicted delays.
* **Queue Management:** The `handle_early_arrival` function manages the patient queue, and prioritizes early arriving patients.

## Files

* `main.py`: Contains the Python code for the AI solution.
* `README.md`: This file, providing project documentation.

## Getting Started

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Shobika-k2004/feature-patient-flow-optimization.git](https://github.com/Shobika-k2004/feature-patient-flow-optimization.git)
    cd feature-patient-flow-optimization
    ```
2.  **Run the Python script:**
    ```bash
    main.py
    ```
3.  Follow the prompts to input patient arrival and appointment details.

## Future Enhancements

* Integrate with a real-world clinic database.
* Implement a more robust patient arrival forecasting model.
* Develop a user interface for clinic staff.
* Add error handling and logging.
* Implement real-time updates for slot allocations.





