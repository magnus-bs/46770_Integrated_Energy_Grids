# README

This repository contains the code and results for the DTU course **46770 – Integrated Energy Grids** course project.

## Structure

All tasks from **Assignment 1 (a–e)** and **Assignment 2 (f–h)** are implemented in the main file:

```text id="yq5w9m"
main.py
```

The following tasks are implemented in separate files:

* **Task i) – Multi-carrier / sector coupling**

```text id="4n5k7t"
Part_2_i_multicarrier.py
```

* **Task j) – Regional case study / experiment**

```text id="z0t4m1"
Part_2_j_casestudy.py
```

## Files

* `main.py`
  Contains the implementation of:

  * Renewable and non-renewable generation optimisation
  * Weather year sensitivity analysis
  * Storage integration
  * Transmission network modelling and DC power flow
  * CO₂ constraint analysis
  * Gas network modelling
  * CO₂ pricing analysis

* `Part_2_i_multicarrier.py`
  Contains the implementation of:

  * Sector coupling between electricity and heating sectors
  * Multi-carrier optimisation analysis

* `Part_2_j_casestudy`
  Contains the implementation of:

  * No nuclear plants in France Scenario

## Requirements

The project is implemented in Python using:

* PyPSA
* pandas
* numpy
* matplotlib

Install dependencies before running the scripts.

## Usage

Run the main project tasks:

```bash id="x7v8c2"
python main.py
```

Run the multi-carrier model:

```bash id="5m2q8r"
python multicarrier.py
```

Run the regional experiment:

```bash id="w3n6k1"
python experiment.py
```
