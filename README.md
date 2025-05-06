# 🛠️ NASA Turbofan Engine Predictive Maintenance using LSTM

This project demonstrates predictive maintenance using the NASA C-MAPSS FD001 dataset. The goal is to predict the **Remaining Useful Life (RUL)** of aircraft engines using an LSTM-based deep learning approach.

---

## 📂 Dataset

Dataset used: [NASA C-MAPSS - FD001 subset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

* **Train File**: Time series sensor data per engine until failure.
* **Test File**: Partial engine life cycles (not till failure).
* **RUL File**: Actual RUL values for test engines.

---

## 🚀 Project Workflow

### 1. 📥 Data Acquisition

* Used `kagglehub` and `kaggle.json` for automatic dataset download.
* Unzipped the files and verified directory structure.

### 2. 🧼 Data Preprocessing

* Assigned meaningful column names for clarity.
* Calculated RUL for each engine in training and test data.
* Merged RUL values from the separate file into the test set.
* Removed constant and quasi-constant features.
* Visualized correlation heatmaps and dropped highly correlated features to avoid redundancy.
* Detected and reported features with >50% outliers.

### 3. 📊 Feature Selection

* Selected features starting with:

  * `op_setting_`
  * `sm_`

### 4. 📈 Sequence Preparation for LSTM

* Converted data to 3D sequences using a **sliding window of 100 time steps** per engine.
* Applied **MinMax Scaling** to normalize the features.

### 5. 🧠 Model Architecture

Built a Sequential LSTM Model:

```text
LSTM (50 units) → Dropout (0.2) → LSTM (50 units) → Dropout (0.2) → Dense (1 unit)
```

* Optimizer: Adam
* Loss: Mean Squared Error (MSE)

### 6. 🏋️ Model Training

* Trained for 100 epochs with a batch size of 16.
* Used 20% of the data for validation.

### 7. 📈 Evaluation & Visualization

* Plotted training and validation loss curves.
* Evaluated on the test set.
* Visualized predicted vs actual RUL values.

---

## 📊 Results

* **Test MSE Loss** is printed after evaluation.
* Line plot shows predicted vs actual RUL to visually assess performance.

---

## 🔧 Future Work

* Hyperparameter tuning for LSTM layers and sequence length.
* Try BiLSTM, CNN-LSTM, or attention-based models.
* Explore ensemble learning or physics-informed models.

---

## 📁 Folder Structure

```
📦 nasa-predictive-maintenance
├── nasa_cmaps/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   ├── RUL_FD001.txt
├── model/
│   ├── model.h5 (optional if saved)
├── README.md
└── your_notebook.ipynb
```

---

## 📚 References

* [NASA C-MAPSS Dataset Info](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
* Saxena, A., Goebel, K., et al. "Turbofan Engine Degradation Simulation Data Set" (NASA Ames)

