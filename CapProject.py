"""
Capstone Project - RateMyProfessor Analysis

Bilal Naseer

"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


# Question 1 
print("Q1")

# Seed RNG
np.random.seed(15971846)

# Load dataset
num_df = pd.read_csv("/Users/bilalnaseer/Documents/Spring '25/PODS/Casptone project/rmpCapstoneNum.csv")

# Rename columns
num_df.columns = [
    "avg_rating", "avg_difficulty", "num_ratings", "pepper",
    "would_take_again_pct", "num_online_ratings", "male", "female"
]

# Filter for professors with 5+ ratings and valid gender
filtered_df = num_df[
    (num_df["num_ratings"] >= 5) &
    ((num_df["male"] == 1) | (num_df["female"] == 1))
]

# Split ratings by gender
male_ratings = filtered_df[filtered_df["male"] == 1]["avg_rating"].dropna()
female_ratings = filtered_df[filtered_df["female"] == 1]["avg_rating"].dropna()

# Welch's t-test
t_stat, p_value = stats.ttest_ind(male_ratings, female_ratings, equal_var=False)

# Stats
male_mean = male_ratings.mean()
female_mean = female_ratings.mean()
male_std = male_ratings.std()
female_std = female_ratings.std()
n_male = len(male_ratings)
n_female = len(female_ratings)

# Confidence interval for difference in means
cm = CompareMeans(DescrStatsW(male_ratings), DescrStatsW(female_ratings))
ci_low, ci_high = cm.tconfint_diff(usevar='unequal', alpha=0.005)

# Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=[male_ratings, female_ratings], palette="Set2")
plt.xticks([0, 1], ["Male", "Female"])
plt.ylabel("Average Rating")
plt.title("Average Ratings by Gender")
plt.tight_layout()
plot_path = "/Users/bilalnaseer/Documents/Spring '25/PODS/Casptone project/q1_gender_bias_boxplot.png"
plt.savefig(plot_path)
plt.close()

# Results dictionary
results_q1 = {
    "n_male": n_male,
    "n_female": n_female,
    "mean_male": male_mean,
    "mean_female": female_mean,
    "std_male": male_std,
    "std_female": female_std,
    "t_stat": t_stat,
    "p_value": p_value,
    "ci_low": ci_low,
    "ci_high": ci_high,
    "boxplot_path": plot_path
}

print(results_q1)
print()


# Question 2
print("Q2")

# Filter: professors with 5+ ratings and no missing rating or num_ratings
df_q2 = num_df[
    (num_df["num_ratings"] >= 5) &
    (~num_df["avg_rating"].isna()) &
    (~num_df["num_ratings"].isna())
]

# Correlation
r_val, p_val = pearsonr(df_q2["num_ratings"], df_q2["avg_rating"])

# Regression setup
X = sm.add_constant(df_q2["num_ratings"])
y = df_q2["avg_rating"]
model = sm.OLS(y, X).fit()
r_squared = model.rsquared
slope = model.params["num_ratings"]
intercept = model.params["const"]

# Plotting
plt.figure(figsize=(8, 5))
sns.scatterplot(x="num_ratings", y="avg_rating", data=df_q2, alpha=0.3, edgecolor=None)
sns.regplot(x="num_ratings", y="avg_rating", data=df_q2, scatter=False, color="red")
plt.xlabel("Number of Ratings (Experience)")
plt.ylabel("Average Teaching Rating")
plt.title("Relationship Between Experience and Teaching Ratings")
plt.tight_layout()
plot_path = "q2_experience_vs_rating.png"  # Saves to your current working folder
plt.savefig(plot_path)
plt.close()

# Output key results
results_q2 = {
    "n": len(df_q2),
    "pearson_r": r_val,
    "p_value": p_val,
    "r_squared": r_squared,
    "slope": slope,
    "intercept": intercept,
    "plot_path": plot_path
}

print(results_q2)
print()


# Question 3
print("Q3")

# Filter: professors with 5+ ratings and no missing avg_rating or avg_difficulty
df_q3 = num_df[
    (num_df["num_ratings"] >= 5) &
    (~num_df["avg_rating"].isna()) &
    (~num_df["avg_difficulty"].isna())
]

# Pearson correlation
r_val, p_val = pearsonr(df_q3["avg_difficulty"], df_q3["avg_rating"])

# Linear regression
X = sm.add_constant(df_q3["avg_difficulty"])
y = df_q3["avg_rating"]
model = sm.OLS(y, X).fit()
r_squared = model.rsquared
slope = model.params["avg_difficulty"]
intercept = model.params["const"]

# Plotting scatter + regression line
plt.figure(figsize=(8, 5))
sns.scatterplot(x="avg_difficulty", y="avg_rating", data=df_q3, alpha=0.3, edgecolor=None)
sns.regplot(x="avg_difficulty", y="avg_rating", data=df_q3, scatter=False, color="red")
plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.title("Relationship Between Difficulty and Teaching Ratings")
plt.tight_layout()
plot_path = "/Users/bilalnaseer/Documents/Spring '25/PODS/Casptone project/q3_difficulty_vs_rating.png"
plt.savefig(plot_path)
plt.close()

# Output key results
results_q3 = {
    "n": len(df_q3),
    "pearson_r": r_val,
    "p_value": p_val,
    "r_squared": r_squared,
    "slope": slope,
    "intercept": intercept,
    "plot_path": plot_path
}

print(results_q3)
print()


# Question 4
print("Q4")

# Filter: Professors with 5+ ratings and valid online data
df_q4 = num_df[
    (num_df["num_ratings"] >= 5) &
    (~num_df["avg_rating"].isna()) &
    (~num_df["num_online_ratings"].isna())
]

df_q4 = df_q4.copy()
df_q4["online_group"] = np.where(df_q4["num_online_ratings"] >= 10, "Many", "Few")

# Splitting the groups
many_online = df_q4[df_q4["online_group"] == "Many"]["avg_rating"].dropna()
few_online = df_q4[df_q4["online_group"] == "Few"]["avg_rating"].dropna()

# Welch's t-test
t_stat, p_val = stats.ttest_ind(many_online, few_online, equal_var=False)

# Stats
mean_many = many_online.mean()
mean_few = few_online.mean()
std_many = many_online.std()
std_few = few_online.std()
n_many = len(many_online)
n_few = len(few_online)

# Confidence interval for difference in means
cm = CompareMeans(DescrStatsW(many_online), DescrStatsW(few_online))
ci_low, ci_high = cm.tconfint_diff(usevar='unequal', alpha=0.005)

# Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x="online_group", y="avg_rating", data=df_q4, hue="online_group", palette="coolwarm", legend=False)
plt.xlabel("Online Teaching Group")
plt.ylabel("Average Rating")
plt.title("Average Ratings: Many vs Few Online Classes")
plt.tight_layout()
plot_path = "/Users/bilalnaseer/Documents/Spring '25/PODS/Casptone project/q4_online_classes_boxplot.png"
plt.savefig(plot_path)
plt.close()

# Output key results
results_q4 = {
    "n_many": n_many,
    "n_few": n_few,
    "mean_many": mean_many,
    "mean_few": mean_few,
    "std_many": std_many,
    "std_few": std_few,
    "t_stat": t_stat,
    "p_value": p_val,
    "ci_low": ci_low,
    "ci_high": ci_high,
    "plot_path": plot_path
}

print(results_q4)
print()


# Question 5
print("Q5")


# Filter: Professors with 5+ ratings and valid would_take_again_pct and avg_rating
df_q5 = num_df[
    (num_df["num_ratings"] >= 5) &
    (~num_df["avg_rating"].isna()) &
    (~num_df["would_take_again_pct"].isna())
]

# Pearson correlation
r_val, p_val = pearsonr(df_q5["would_take_again_pct"], df_q5["avg_rating"])

# Linear regression
X = sm.add_constant(df_q5["would_take_again_pct"])
y = df_q5["avg_rating"]
model = sm.OLS(y, X).fit()
slope = model.params["would_take_again_pct"]
intercept = model.params["const"]
r_squared = model.rsquared

# Plotting scatter with regression line
plt.figure(figsize=(8, 5))
sns.scatterplot(x="would_take_again_pct", y="avg_rating", data=df_q5, alpha=0.3, edgecolor=None)
sns.regplot(x="would_take_again_pct", y="avg_rating", data=df_q5, scatter=False, color="red")
plt.xlabel("Would Take Again Percentage")
plt.ylabel("Average Teaching Rating")
plt.title("Relationship Between 'Would Take Again' % and Ratings")
plt.tight_layout()
plot_path = "/Users/bilalnaseer/Documents/Spring '25/PODS/Casptone project/q5_take_again_vs_rating.png"
plt.savefig(plot_path)
#plt.show()  
plt.close()

# Output key results
results_q5 = {
    "n": len(df_q5),
    "pearson_r": r_val,
    "p_value": p_val,
    "slope": slope,
    "intercept": intercept,
    "r_squared": r_squared,
    "plot_path": plot_path
}

print(results_q5)
print()


# Question 6
print("Q6")

# Filter: Professors with 5+ ratings and non-missing values for avg_rating and pepper
df_q6 = num_df[
    (num_df["num_ratings"] >= 5) &
    (~num_df["avg_rating"].isna()) &
    (~num_df["pepper"].isna())
].copy()  # avoids SettingWithCopyWarning

# Converting pepper to boolean
df_q6["pepper"] = df_q6["pepper"].astype(bool)

# Spliting groups
peppered = df_q6[df_q6["pepper"] == True]["avg_rating"]
non_peppered = df_q6[df_q6["pepper"] == False]["avg_rating"]

# Welch’s t-test
from scipy import stats
t_stat, p_val = stats.ttest_ind(peppered, non_peppered, equal_var=False)

# Stats
mean_pepper = peppered.mean()
mean_no_pepper = non_peppered.mean()
std_pepper = peppered.std()
std_no_pepper = non_peppered.std()
n_pepper = len(peppered)
n_no_pepper = len(non_peppered)

# Confidence interval
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
cm = CompareMeans(DescrStatsW(peppered), DescrStatsW(non_peppered))
ci_low, ci_high = cm.tconfint_diff(usevar='unequal', alpha=0.005)

# Plot
df_q6["pepper_label"] = np.where(df_q6["pepper"], "Has Pepper", "No Pepper")
plt.figure(figsize=(8, 5))
sns.boxplot(x="pepper_label", y="avg_rating", hue="pepper_label", data=df_q6, palette="pastel", legend=False)
plt.xlabel("Chili Pepper Status")
plt.ylabel("Average Rating")
plt.title("Average Ratings by Chili Pepper Status")
plt.tight_layout()
plot_path = "/Users/bilalnaseer/Documents/Spring '25/PODS/Casptone project/q6_chili_pepper_boxplot.png"
plt.savefig(plot_path)
plt.show()
plt.close()

# Output results
results_q6 = {
    "n_pepper": n_pepper,
    "n_no_pepper": n_no_pepper,
    "mean_pepper": mean_pepper,
    "mean_no_pepper": mean_no_pepper,
    "std_pepper": std_pepper,
    "std_no_pepper": std_no_pepper,
    "t_stat": t_stat,
    "p_value": p_val,
    "ci_low": ci_low,
    "ci_high": ci_high,
    "plot_path": plot_path
}

print(results_q6)
print()


# Question 7
print("Q7")

# Filter: Professors with 5+ ratings and valid avg_rating and avg_difficulty
df_q7 = num_df[
    (num_df["num_ratings"] >= 5) &
    (~num_df["avg_rating"].isna()) &
    (~num_df["avg_difficulty"].isna())
]

# Setting up variables
X = sm.add_constant(df_q7["avg_difficulty"])
y = df_q7["avg_rating"]

# Linear regression model
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Calculating RMSE
rmse = np.sqrt(np.mean((y - predictions) ** 2))

# Getting stats
slope = model.params["avg_difficulty"]
intercept = model.params["const"]
r_squared = model.rsquared

# Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x="avg_difficulty", y="avg_rating", data=df_q7, alpha=0.3, edgecolor=None)
sns.regplot(x="avg_difficulty", y="avg_rating", data=df_q7, scatter=False, color="red")
plt.xlabel("Average Difficulty")
plt.ylabel("Average Rating")
plt.title("Linear Regression: Predicting Rating from Difficulty")
plt.tight_layout()
plot_path = "/Users/bilalnaseer/Documents/Spring '25/PODS/Casptone project/q7_regression_difficulty_rating.png"
plt.savefig(plot_path)
plt.show()
plt.close()

# Output key results
results_q7 = {
    "n": len(df_q7),
    "slope": slope,
    "intercept": intercept,
    "r_squared": r_squared,
    "rmse": rmse,
    "plot_path": plot_path
}

print(results_q7)
print()



# Question 8
print("Q8")

# Filter: Professors with 5+ ratings and complete data for predictors
df_q8 = num_df[
    (num_df["num_ratings"] >= 5) &
    (~num_df["avg_rating"].isna()) &
    (~num_df["avg_difficulty"].isna()) &
    (~num_df["would_take_again_pct"].isna()) &
    (~num_df["num_online_ratings"].isna()) &
    (~num_df["pepper"].isna())
].copy()

# Converting 'pepper' to binary
df_q8["pepper"] = df_q8["pepper"].astype(bool).astype(int)

# Setting up predictors and target
X = df_q8[["avg_difficulty", "would_take_again_pct", "num_online_ratings", "pepper"]]
y = df_q8["avg_rating"]
X_const = sm.add_constant(X)

# Regression model
model = sm.OLS(y, X_const).fit()
predictions = model.predict(X_const)

# R-square and RMSE
r_squared = model.rsquared
rmse = np.sqrt(np.mean((y - predictions) ** 2))
betas = model.params.to_dict()

# Plotting Predicted vs Actual
plt.figure(figsize=(8, 5))
sns.scatterplot(x=predictions, y=y, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel("Predicted Rating")
plt.ylabel("Actual Rating")
plt.title("Predicted vs Actual Ratings (Full Model)")
plt.tight_layout()
plot_path = "/Users/bilalnaseer/Documents/Spring '25/PODS/Casptone project/q8_full_model_rating_prediction.png"
plt.savefig(plot_path)
plt.close()

# Output key results
results_q8 = {
    "n": len(df_q8),
    "r_squared": r_squared,
    "rmse": rmse,
    "betas": betas,
    "plot_path": plot_path
}

print(results_q8)
print()


# Question 9
print("Q9")

# Filter: Professors with 5+ ratings and no missing values for avg_rating and pepper
df_q9 = num_df[
    (num_df["num_ratings"] >= 5) &
    (~num_df["avg_rating"].isna()) &
    (~num_df["pepper"].isna())
].copy()

# Converting 'pepper' to binary
df_q9["pepper"] = df_q9["pepper"].astype(bool).astype(int)

# Class balance
pepper_counts = df_q9["pepper"].value_counts(normalize=True)

# Defining X and y
X = df_q9[["avg_rating"]]
y = df_q9["pepper"]

# Train-test split with stratification for class imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15971846)


# Fitting logistic regression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_prob = clf.predict_proba(X_test)[:, 1]

# AUROC
auc_score = roc_auc_score(y_test, y_pred_prob)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUROC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Predicting Chili Pepper from Average Rating")
plt.legend(loc="lower right")
plt.tight_layout()
plot_path = "/Users/bilalnaseer/Documents/Spring '25/PODS/Casptone project/q9_roc_curve_pepper_prediction.png"
plt.savefig(plot_path)
plt.close()

# Coefficients
coef = clf.coef_[0][0]
intercept = clf.intercept_[0]

# Output key results
results_q9 = {
    "n": len(df_q9),
    "class_balance": pepper_counts.to_dict(),
    "coefficient": coef,
    "intercept": intercept,
    "auroc": auc_score,
    "roc_curve_path": plot_path
}

print(results_q9)
print()


# Question 10
print("Q10")

# Filter: Professors with 5+ ratings and complete predictor data
df_q10 = num_df[
    (num_df["num_ratings"] >= 5) &
    (~num_df["pepper"].isna()) &
    (~num_df["avg_rating"].isna()) &
    (~num_df["avg_difficulty"].isna()) &
    (~num_df["would_take_again_pct"].isna()) &
    (~num_df["num_online_ratings"].isna())
].copy()

# Converting 'pepper' to binary
df_q10["pepper"] = df_q10["pepper"].astype(bool).astype(int)

# Setting up predictors and outcome
X = df_q10[["avg_rating", "avg_difficulty", "would_take_again_pct", "num_online_ratings"]]
y = df_q10["pepper"]

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15971846)

# Logistic regression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred_prob = clf.predict_proba(X_test)[:, 1]

# AUROC
auc_score = roc_auc_score(y_test, y_pred_prob)

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUROC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Predicting Chili Pepper from All Factors")
plt.legend(loc="lower right")
plt.tight_layout()
plot_path = "/Users/bilalnaseer/Documents/Spring '25/PODS/Casptone project/q10_roc_curve_pepper_all_features.png"
plt.savefig(plot_path)
plt.close()

# Model coefficients
coef_dict = dict(zip(X.columns, clf.coef_[0]))
intercept = clf.intercept_[0]

# Class balance
pepper_counts_q10 = y.value_counts(normalize=True)

# Output key results 
results_q10 = {
    "n": len(df_q10),
    "class_balance": pepper_counts_q10.to_dict(),
    "coefficients": coef_dict,
    "intercept": intercept,
    "auroc": auc_score,
    "roc_curve_path": plot_path
}

print(results_q10)
print()


# Extra Credit
print("Extra Credit")

# Loading qualitative data
qual_df = pd.read_csv("/Users/bilalnaseer/Documents/Spring '25/PODS/Casptone project/rmpCapstoneQual.csv", names=["major", "university", "state"])

# Merging with numeric data 
merged_df = pd.concat([num_df, qual_df], axis=1)

# Filtering: Professors with 5+ ratings and non-missing avg_rating
filtered_ec_df = merged_df[
    (merged_df["num_ratings"] >= 5) &
    (~merged_df["avg_rating"].isna()) &
    (~merged_df["state"].isna())
]

# Counting top 5 states by number of professors
top_states = filtered_ec_df["state"].value_counts().head(5).index.tolist()

# Filtering to only these top states
df_top_states = filtered_ec_df[filtered_ec_df["state"].isin(top_states)]

# Performing Welch’s ANOVA (using scipy.stats — for simplicity, compare with t-test logic pairwise or via statsmodels)
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Fitting model for ANOVA
model = ols('avg_rating ~ C(state)', data=df_top_states).fit()
anova_results = sm.stats.anova_lm(model, typ=2)  # ANOVA table

# Boxplot comparing states
plt.figure(figsize=(8, 5))
sns.boxplot(x="state", y="avg_rating", hue="state", data=df_top_states, palette="Set3", legend=False)
plt.xlabel("U.S. State")
plt.ylabel("Average Rating")
plt.title("Average Ratings Across Top 5 U.S. States")
plt.tight_layout()
plot_path = "/Users/bilalnaseer/Documents/Spring '25/PODS/Casptone project/extracredit_state_rating_boxplot.png"
plt.savefig(plot_path)
plt.close()


# Output key results
print("Welch’s ANOVA Results (via ANOVA table):")
print(anova_results)














