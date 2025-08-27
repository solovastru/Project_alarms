import pandas as pd
import numpy as np


class DataProcessor:
    def __init__(self, thresholds, dataset):
        self.thresholds = thresholds
    

    def discretize_vars(self, df):
        # --- OEE ---
        if "OEE" in self.thresholds:
            low_t = self.thresholds["OEE"]["low_t"]
            high_t = self.thresholds["OEE"]["high_t"]

            df["OEE"] = pd.cut(
                df["OEE"],
                bins=[0, low_t, high_t, float("inf")],
                labels=["Low", "Medium", "High"]
            )

        # --- SPARE_COST ---
        if "SPARE_COST" in self.thresholds:
            low_t = self.thresholds["SPARE_COST"]["low_t"]
            high_t = self.thresholds["SPARE_COST"]["high_t"]

            df["SPARE_COST"] = pd.cut(
                df["SPARE_COST"],
                bins=[0.0, low_t, high_t, float("inf")],
                labels=["Low", "Medium", "High"],
                include_lowest=True
            )

        return df
    
# thresholds for OEE are determined on the OEE benchmark, the ones for Spare_cost are done based on the equal-width binning technique
thresholds_cat = {
    "OEE": {'low_t': 60, 'high_t': 85},
    'SPARE_COST': {'low_t': 300,  'high_t': 900},
}

file_path = "C:\\Users\\roxan\\Desktop\\TU_project\\Alarms-dataset.xlsx"
df = pd.read_excel(file_path)



processor = DataProcessor(thresholds_cat, df)
df = processor.discretize_vars(df)

output_path = "C:\\Users\\roxan\\Desktop\\TU_project\\discretized_alarms.xlsx"
df.to_excel(output_path, index=False)

mappging = {"Very_High": 87.5, "High": 62.5, "Medium": 37.5, "Low": 12.5}