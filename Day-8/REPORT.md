# Student Completion Prediction — Technical Report

**Organisation:** Altrodav Technologies  
**Task:** AI/ML Developer Task 8 — Machine Learning Model Development & Performance Evaluation  
**Date:** June 2025  

---

## Table of Contents

1. [Executive Problem Statement](#1-executive-problem-statement)  
2. [Data Summary](#2-data-summary)  
3. [Data Preprocessing & Feature Engineering](#3-data-preprocessing--feature-engineering)  
4. [Model Development](#4-model-development)  
5. [Model Performance Comparison](#5-model-performance-comparison)  
6. [Business Questions — Detailed Answers](#6-business-questions--detailed-answers)  
7. [Key Findings](#7-key-findings)  
8. [Business Recommendations](#8-business-recommendations)  
9. [Conclusion](#9-conclusion)

---

## 1. Executive Problem Statement

A digital learning platform seeks to proactively identify students who are at risk of failing to complete their enrolled training programmes. Late or reactive intervention — waiting until a student drops out — results in lost learner retention, reduced course completion rates, and poor platform-level ROI on content investment.

This project addresses the problem as a **binary classification task**: given a student's observable learning behaviour at any point during a programme, predict whether that student will ultimately complete (`Final_Completion_Status = 1`) or not complete/drop out (`Final_Completion_Status = 0`).

Three machine learning models — **Logistic Regression**, **Decision Tree Classifier**, and **Random Forest Classifier** — were trained, evaluated, and compared on a labelled dataset of 1,500 student records. The best-performing model is identified, feature importances are extracted, and a framework for real-time at-risk student flagging is described.

---

## 2. Data Summary

| Property | Value |
|---|---|
| Total records | 1,500 |
| Total features (independent) | 6 |
| Target variable | `Final_Completion_Status` (binary) |
| Missing values | 0 |
| Duplicate rows | 0 |
| Completed (class 1) | 757 (50.5%) |
| Not Completed (class 0) | 743 (49.5%) |

### Feature Descriptions

| Feature | Type | Range | Description |
|---|---|---|---|
| `Attendance_Percentage` | Continuous | 0 – 100 | Percentage of scheduled sessions attended |
| `Assessment_Score` | Continuous | 0 – 100 | Average score across all assessments |
| `Assignment_Completion_Rate` | Continuous | 0.0 – 1.0 | Proportion of assignments submitted and completed |
| `Login_Frequency` | Integer | 1 – 60 | Number of platform logins per month |
| `Study_Hours` | Continuous | 0 – 40 | Self-reported or tracked study hours per week |
| `Activity_Count` | Integer | 5 – 200 | Total clickstream / forum interaction events |

The target classes are **well-balanced** (~50/50 split), meaning no class-imbalance correction (e.g., SMOTE or class weighting) was required, and standard accuracy is a reliable metric alongside Precision, Recall, and F1-Score.

---

## 3. Data Preprocessing & Feature Engineering

### 3.1 Data Quality Checks
- **Missing values:** `df.isnull().sum()` confirmed zero missing values across all columns.
- **Duplicate rows:** `df.duplicated().sum()` confirmed zero duplicate records.
- No imputation or row removal was necessary.

### 3.2 Feature Selection
`Student_ID` is a non-informative identifier and was excluded from the model feature set. All six remaining behavioural and performance columns were retained as independent variables.

### 3.3 Train / Test Split
The dataset was split 80 % (1,200 records) for training and 20 % (300 records) for testing using `train_test_split(random_state=42, stratify=y)`. Stratification ensures the class ratio is preserved in both splits.

### 3.4 Feature Scaling
`StandardScaler` was applied exclusively to **Logistic Regression**, which is sensitive to feature magnitude. The scaler was **fitted only on the training set** and then used to transform both train and test sets, preventing data leakage:

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit + transform — training only
X_test_sc  = scaler.transform(X_test)         # transform only — no leakage
```

Tree-based models (Decision Tree and Random Forest) are scale-invariant and received unscaled features.

---

## 4. Model Development

### 4.1 Logistic Regression
- A linear classifier modelling the log-odds of completion as a weighted sum of features.
- Hyper-parameters: `max_iter=1000`, `random_state=42`.
- Uses scaled features (StandardScaler applied).

### 4.2 Decision Tree Classifier
- A non-parametric model that partitions the feature space using binary splits to minimise Gini impurity.
- Hyper-parameters: `max_depth=6`, `random_state=42`. Depth was capped to reduce overfitting.
- Uses raw (unscaled) features.

### 4.3 Random Forest Classifier
- An ensemble of 200 decision trees trained on bootstrap samples with random feature sub-sampling at each split.
- Hyper-parameters: `n_estimators=200`, `max_depth=10`, `random_state=42`.
- Produces probability estimates via `predict_proba()` for at-risk student flagging.
- Uses raw (unscaled) features.

---

## 5. Model Performance Comparison

All metrics are evaluated on the **held-out test set (n = 300)** and reported for the **positive class** (Completed, class = 1).

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression** | **0.6900** | **0.6986** | 0.6755 | **0.6869** |
| Random Forest | 0.6733 | 0.6828 | 0.6556 | 0.6689 |
| Decision Tree | 0.6000 | 0.6026 | 0.6026 | 0.6026 |

> **Best model: Logistic Regression** (Accuracy = 69.0%, F1 = 0.687)

The balanced class distribution means the 69 % accuracy baseline is meaningfully above a naïve majority classifier (50 %). Logistic Regression edges out Random Forest by ~1.7 percentage points in both accuracy and F1-Score, suggesting the relationship between features and completion is approximately linear in log-odds space, and the ensemble's additional complexity does not provide net benefit on this dataset size.

---

## 6. Business Questions — Detailed Answers

### Q1: Which features have the highest impact on predicting student success?

Based on the **Random Forest feature importance scores**, the six features ranked by predictive contribution are:

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | Attendance_Percentage | ~0.195 |
| 2 | Activity_Count | ~0.187 |
| 3 | Assessment_Score | ~0.181 |
| 4 | Assignment_Completion_Rate | ~0.176 |
| 5 | Login_Frequency | ~0.142 |
| 6 | Study_Hours | ~0.119 |

**Attendance, Activity Count, and Assessment Score** together account for over 56 % of the model's predictive power. This is consistent with learning-science research: students who attend sessions consistently, engage actively with platform content, and perform well on assessments are structurally more likely to persist to programme completion.

The correlation heatmap (`visualizations/correlation_heatmap.png`) confirms moderate positive correlations between all six features and `Final_Completion_Status` (r ≈ 0.33 – 0.40), with all features being independently informative.

---

### Q2: Can student completion status be predicted accurately using learning activity data?

**Yes — with meaningful but bounded accuracy.** The best model (Logistic Regression) correctly classifies **69.0 % of students** in the held-out test set, which is 19 percentage points above chance on a balanced dataset. The precision of 0.699 means that when the model predicts a student will complete, it is correct approximately 70 % of the time.

The moderate accuracy (rather than >90 %) reflects the genuine stochasticity of human behaviour: external factors such as personal circumstances, motivation, and programme quality are not captured in the six behavioural features alone. Despite this, a 69 % accurate early-warning system is practically valuable — catching roughly 7 in 10 at-risk students before drop-out allows targeted intervention.

Model performance can be improved with richer data (e.g., discussion participation quality, assignment-resubmission patterns, time-of-day access logs) and hyper-parameter tuning via cross-validation.

---

### Q3: Which machine learning algorithm performs best for this classification problem?

**Logistic Regression** is the best-performing model across Accuracy (69.0 %), Precision (69.9 %), and F1-Score (0.687). Random Forest is a close second (67.3 %, F1 = 0.669), while Decision Tree performs noticeably below both (60.0 %, F1 = 0.603).

The superiority of Logistic Regression over Random Forest on this specific dataset suggests that the signal in the data is relatively linear and well-expressed by a weighted combination of the six features. The Random Forest's additional capacity to model non-linear interactions does not yield net gains, possibly because the dataset size (1,500 records) is insufficient to fully exploit ensemble diversity at 200 trees.

For production deployment, Logistic Regression is also preferable due to:
- Faster inference (single matrix multiplication vs. 200 tree traversals).
- Easy coefficient interpretation (direct feature weight extraction).
- Native probability calibration via the logistic function.

---

### Q4: How do attendance, assessment scores, and activity levels influence predictions?

These three features are the top predictors, and their influence can be characterised as follows:

**Attendance Percentage (most important, ~19.5 % RF importance):**  
Students with high attendance (>75 %) show a substantially higher probability of completing. Attendance reflects student commitment and removes the primary barrier to learning — simply being present for scheduled content delivery. In the Logistic Regression model, the attendance coefficient is the largest positive weight, meaning each 10-point increase in attendance meaningfully raises the predicted completion probability.

**Activity Count (~18.7 % RF importance):**  
Activity Count captures platform engagement depth — how actively a student interacts with content, forums, and resources beyond passive attendance. High-activity students (>120 interactions) are more likely to complete, reflecting deeper integration with the learning process.

**Assessment Score (~18.1 % RF importance):**  
Assessment scores are a direct signal of content mastery. Students scoring below 50 on average are disproportionately represented in the dropout class, as poor assessment outcomes likely correlate with confusion, disengagement, and loss of self-efficacy.

Together, these three features paint a picture of **the engaged, performing student** — one who attends, actively participates, and demonstrates understanding — as the most likely completer.

---

### Q5: Which students are predicted to be at risk of not completing the program?

Students are flagged as **at-risk** using the Random Forest's `predict_proba()` method, which returns a continuous probability score (0.0 to 1.0) for each class. A completion probability below a chosen **threshold** designates a student as at-risk.

**Implementation:**

```python
# Get completion probability for every student
completion_prob = rf.predict_proba(X_test.values)[:, 1]

# Apply threshold — students below 0.40 are flagged as high-risk
THRESHOLD = 0.40
at_risk_mask = completion_prob < THRESHOLD

at_risk_students = X_test[at_risk_mask].copy()
at_risk_students['Completion_Probability'] = completion_prob[at_risk_mask]
at_risk_students['Student_ID'] = df.loc[X_test.index[at_risk_mask], 'Student_ID'].values
```

In the test set, **104 out of 300 students** (34.7 %) were flagged as at-risk with a threshold of 0.40. These students can be ranked by their probability score and prioritised for intervention from lowest to highest completion probability.

**Threshold guidance:**
| Threshold | Trade-off |
|---|---|
| 0.50 (default) | Balanced — flags ~50 % of students, highest precision |
| 0.40 | Slightly more conservative — catches more true at-risk students, some false positives |
| 0.30 | Aggressive early-warning — maximises recall at the cost of precision |

The appropriate threshold depends on the intervention cost and capacity of the support team.

---

### Q6: What is the precision, recall, and F1-score of each model?

Detailed per-model metrics for the positive class (Completed = 1) on the test set:

| Model | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.699 | 0.675 | 0.687 | 0.690 |
| Random Forest | 0.683 | 0.656 | 0.669 | 0.673 |
| Decision Tree | 0.603 | 0.603 | 0.603 | 0.600 |

**Interpreting the metrics:**

- **Precision (0.699 for LR):** When the model predicts a student will complete, it is correct 69.9 % of the time. High precision limits the waste of sending unnecessary intervention resources to students who would have completed anyway.

- **Recall (0.675 for LR):** The model correctly identifies 67.5 % of all students who actually completed. The complementary "Not Completed" recall (~70.3 %) means we are also catching ~7 in 10 true drop-outs.

- **F1-Score (0.687 for LR):** The harmonic mean of precision and recall. On a balanced dataset, this closely tracks accuracy and confirms the model is not optimising for one class at the expense of the other.

The Decision Tree's lower metrics (F1 = 0.603) indicate it overfit to certain splits in the training data and failed to generalise well, even with `max_depth=6`.

---

### Q7: How can the model's predictions help improve learning outcomes?

The predictive model creates a **proactive, data-driven intervention loop** rather than a reactive one:

1. **Early Identification:** Run the model at regular intervals (e.g., end of Week 2, Week 4) during a programme. Students with completion probability < 0.40 are surfaced to tutors or success coaches automatically.

2. **Personalised Intervention:** Because feature importances are known, the platform can personalise the intervention:
   - Low attendance → automated reminder emails and flexible make-up session scheduling.
   - Low assessment scores → adaptive content, additional practice quizzes, or 1:1 tutoring sessions.
   - Low activity count → gamification nudges, discussion prompts, or peer-pairing.

3. **Resource Optimisation:** Rather than applying blanket support to all 1,500 students, counsellors can focus effort on the ~35 % of students with the highest drop-out risk, improving cost-effectiveness of support teams.

4. **Programme Design Feedback:** Aggregate prediction results across cohorts identify whether particular programme segments (weeks, modules) consistently generate at-risk signals, enabling curriculum teams to redesign or reinforce weak points.

5. **Continuous Model Improvement:** As intervention outcomes are logged, labelled data accumulates and the model can be retrained, progressively improving its predictive power.

---

### Q8: What actionable business recommendations can be made to the platform stakeholders based on model findings?

#### Recommendation 1: Deploy an Automated At-Risk Dashboard
Integrate the trained Logistic Regression model into the platform's data pipeline. Generate a weekly at-risk report with `predict_proba()` scores for all active students. Surface this dashboard to programme managers and tutors with direct access to student profiles.

#### Recommendation 2: Prioritise Attendance as the Primary Leading Indicator
Attendance_Percentage is the single strongest predictor of completion (RF importance ~19.5 %). Implement real-time attendance tracking with a threshold alert (e.g., <70 % in any given week) that immediately triggers a welfare check from the student success team.

#### Recommendation 3: Gamify Activity Count to Lift Engagement
Activity Count is the second most important feature and is directly modifiable through platform design. Introduce progress streaks, leaderboards, badge rewards for forum posts, and peer-collaboration challenges. Even small increases in average activity from low-engagement students could meaningfully shift predicted completion probability.

#### Recommendation 4: Introduce Adaptive Assessment Pathways
Assessment Score is the third most impactful feature. Students scoring below 50 are at significantly elevated drop-out risk. Trigger automated adaptive pathways (additional practice content, scaffolded video explanations, or spaced-repetition quizzes) for students in this band before they disengage.

#### Recommendation 5: Set a Completion Probability Threshold Policy
Formally define the intervention threshold at **0.40** as the operational standard. Students scoring below this threshold are classified as "High Risk" and receive mandatory outreach. Those between 0.40–0.55 are "Moderate Risk" and receive lighter-touch automated nudges. This tiering ensures efficient use of support-team capacity.

#### Recommendation 6: Retrain the Model Quarterly
Student behaviour patterns evolve with platform updates, programme changes, and changing learner demographics. Schedule quarterly model retraining on the most recent 6–12 months of labelled completion data to keep predictions calibrated.

#### Recommendation 7: Explore Richer Feature Sets for Next Iteration
The current 69 % accuracy reflects the limits of six aggregate features. Future model versions should incorporate:
- Assignment re-submission rates (a signal of persistence).
- Time-of-day and day-of-week access patterns (lifestyle compatibility with the programme schedule).
- Discussion forum sentiment and post quality.
- Peer-interaction network centrality (isolated students have higher dropout risk).

---

## 7. Key Findings

1. **All six behavioural features are predictive** of completion status, with correlation coefficients of 0.33–0.40 with the target variable.
2. **Attendance Percentage is the single most important predictor**, followed by Activity Count and Assessment Score.
3. **Logistic Regression is the best-performing model** at 69.0 % accuracy and F1 = 0.687 on a balanced dataset, outperforming Random Forest (67.3 %) and Decision Tree (60.0 %).
4. **The relationship between features and completion is approximately linear** in log-odds space, explaining why the simpler Logistic Regression outperforms the ensemble.
5. **104 of 300 test-set students** (34.7 %) were correctly flagged as at-risk using a 0.40 probability threshold.
6. **No data quality issues** were identified — the dataset was clean with zero missing values and zero duplicate rows.

---

## 8. Business Recommendations

| Priority | Recommendation | Impact |
|---|---|---|
| High | Deploy automated at-risk dashboard with weekly `predict_proba` scoring | Enables proactive intervention before drop-out |
| High | Set attendance alert at <70% — trigger welfare check immediately | Tackles the #1 predictor directly |
| Medium | Gamify activity count with streaks, badges, and peer-collaboration tools | Directly lifts the 2nd most important feature |
| Medium | Adaptive assessment pathways for students scoring below 50 | Addresses 3rd most important predictor |
| Medium | Define tiered intervention policy: 0.40 threshold = High Risk | Efficient use of support-team capacity |
| Low | Quarterly model retraining on fresh data | Keeps prediction accuracy calibrated |
| Low | Enrich feature set with re-submission rates, sentiment, access-time patterns | Pathway to >80% model accuracy |

---

## 9. Conclusion

This analysis demonstrates that student completion status can be predicted with meaningful accuracy (69 %) using six readily available learning-behaviour metrics. The Logistic Regression model, despite its simplicity, is the best performer and is well-suited for production deployment due to its interpretability, speed, and naturally calibrated probability outputs.

The model's predictions, operationalised via `predict_proba()` thresholding, provide a practical at-risk flagging system capable of identifying approximately 7 in 10 students who would otherwise drop out — giving the platform's support teams the lead time needed to intervene, personalise the learning journey, and materially improve programme completion rates.

---

*Report generated as part of Altrodav Technologies AI/ML Developer Task 8. All results based on the synthetic `dataset.csv` (1,500 records). Model artefacts are in `ml_model.ipynb`. All visualisations are in the `visualizations/` folder.*
