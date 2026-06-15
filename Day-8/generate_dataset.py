import numpy as np
import pandas as pd

np.random.seed(42)
N = 1500


# 1. Generate base completion probability latent variable
latent = np.random.normal(0, 1, N)


# 2. Build features correlated with the latent variable

# Attendance_Percentage  (0–100)
attendance = np.clip(50 + 20 * latent + np.random.normal(0, 8, N), 0, 100)

# Assessment_Score  (0–100)
assessment = np.clip(50 + 18 * latent + np.random.normal(0, 9, N), 0, 100)

# Assignment_Completion_Rate  (0.0–1.0)
assignment_rate = np.clip(0.5 + 0.2 * latent + np.random.normal(0, 0.1, N), 0.0, 1.0)

# Login_Frequency  (integer, 1–60 per month)
login_freq = np.clip(
    np.round(20 + 8 * latent + np.random.normal(0, 4, N)).astype(int), 1, 60
)

# Study_Hours  (hours per week, 0–40)
study_hours = np.clip(10 + 4 * latent + np.random.normal(0, 3, N), 0, 40)

# Activity_Count  (integer, 5–200 interactions)
activity = np.clip(
    np.round(80 + 30 * latent + np.random.normal(0, 15, N)).astype(int), 5, 200
)


# 3. Derive binary target from a logistic function over the latent
prob = 1 / (1 + np.exp(-1.2 * latent))
final_status = (np.random.uniform(0, 1, N) < prob).astype(int)


# 4. Assemble DataFrame
df = pd.DataFrame({
    "Student_ID":                  range(1001, 1001 + N),
    "Attendance_Percentage":       np.round(attendance, 2),
    "Assessment_Score":            np.round(assessment, 2),
    "Assignment_Completion_Rate":  np.round(assignment_rate, 4),
    "Login_Frequency":             login_freq,
    "Study_Hours":                 np.round(study_hours, 2),
    "Activity_Count":              activity,
    "Final_Completion_Status":     final_status,
})


# 5. Validate and save
assert len(df) == 1500, "Row count mismatch"
assert df.isnull().sum().sum() == 0, "Unexpected nulls"
print(df.head())
print("\nShape:", df.shape)
print("\nClass distribution:\n", df["Final_Completion_Status"].value_counts())
print("\nCorrelations with target:")
print(df.corr()["Final_Completion_Status"].sort_values(ascending=False))

df.to_csv("dataset.csv", index=False)
print("\n✅  dataset.csv written successfully.")
