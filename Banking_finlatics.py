# '''Project Banking'''

# #Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# #Loading Dataset

df = pd.read_csv(r'C:\Users\gsour\Documents\Finlatics Python\Banking\banking_data.csv')

# #Examining the data

# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)

# print(df.head(10))
# print(df.tail(10))
# print(df.info())
# print(df.shape)
# print(df.columns)
# print(df['marital'].value_counts())
# print(df['marital_status'].value_counts())
# df.drop(columns=['marital'],inplace=True)
# print(df.columns)

# '''What is the distribution of age among the clients?'''

# mean = df['age'].mean()
# median = df['age'].median()
# mode = df['age'].mode()[0]
# max = df['age'].max()
# min = df['age'].min()
# std_deviation = df['age'].std()

# plt.figure(figsize=(6,4))
# sns.boxplot(x=df['age'],color="skyblue")
# plt.xlabel("Age")
# plt.title("Box Plot of Age")
# plt.show()

# print("Age distribution -\nMean:", round(mean,3))
# print("Median:", median)
# print("Mode:",mode)
# print("Maximum value:", max)
# print("Minimum value:", min)
# print("Standard deviation:", round(std_deviation,3))

# if mean>median>mode:
#     print("The distribution is positive skewed.")
# elif mean<median<mode:
#     print("The distribution is negative skewed.")
# elif mean == median == mode:
#     print("The distribution is symmetrical.")

# '''How does the job type vary among the clients?'''

# job_counts = df['job'].value_counts()
# print("Frequency count:", job_counts)

# job_percentage = df['job'].value_counts(normalize=True) * 100
# print("\nRelative frequency (percentage):\n", job_percentage)

# unique_jobs = df['job'].nunique()
# print("\nNumber of unique job types:", unique_jobs)

# job_mode = df['job'].mode()[0]
# print("\nMost common job type (Mode):", job_mode)

# #Pie chart for job distribution
# plt.figure(figsize = (6,6))
# job_counts.plot.pie(autopct= '%1.1f%%', startangle = 90, color = sns.color_palette("viridis"))
# plt.title("Job Type Distribution")
# plt.ylabel('')
# plt.show()

# '''What is the marital status distribution of the clients?'''

# marital_counts = df['marital_status'].value_counts()
# print('Frequency count:', marital_counts)

# marital_percentage = df['marital_status'].value_counts(normalize=True)*100
# print("\nRelative frequency (percentage):\n", round(marital_percentage,3))

# #Pie Chart

# plt.figure(figsize = (6,6))
# marital_counts.plot.pie(autopct='%1.1f%%', startangle=90, color=sns.color_palette("viridis"))
# plt.title("Marital Status Distribution")
# plt.ylabel('')
# plt.show()

# '''What is the level of education among the clients?'''

# edu_counts = df['education'].value_counts()
# print(edu_counts)

# edu_percentage = df['education'].value_counts(normalize=True)*100
# print(edu_percentage)

# #Bar Plot
# plt.figure(figsize=(8,5))
# sns.countplot(x='education', data=df, order=edu_counts.index, palette="viridis")
# plt.ylabel('Frequency')
# plt.title('Education Level Distribution')
# plt.show()

# '''What proportion of clients have credit in default?'''

# default_counts = df['default'].value_counts()
# print(default_counts)

# default_percentage = df['default'].value_counts(normalize=True)*100
# yes_percentage = default_percentage.get('yes',0)
# print(f"Proportion of clients that have credit in default is {yes_percentage:.2f}%.")

# '''What is the distribution of average yearly balance among the clients?'''

# mean_balance = df['balance'].mean()
# median_balance = df['balance'].median()
# mode_balance = df['balance'].mode()[0]
# max_balance = df['balance'].max()
# min_balance = df['balance'].min()
# std_deviation_balance = df['balance'].std()

# plt.figure(figsize=(6,4))
# sns.boxplot(x=df['balance'],color="skyblue")
# plt.xlabel("Average Yearly Balance")
# plt.title("Box Plot of Average Yearly Balance")
# plt.show()

# print("Average Yearly Balance Distribution -\nMean:", round(mean_balance,3))
# print("Median:", median_balance)
# print("Mode:",mode_balance)
# print("Maximum value:", max_balance)
# print("Minimum value:", min_balance)
# print("Standard deviation:", round(std_deviation_balance,3))

# if mean_balance>median_balance>mode_balance:
#     print("The distribution is positive skewed.")
# elif mean_balance<median_balance<mode_balance:
#     print("The distribution is negative skewed.")
# elif mean_balance == median_balance == mode_balance:
#     print("The distribution is symmetrical.")

# '''How many clients have housing loans?'''

# housing_counts = df['housing'].value_counts()
# print(housing_counts)

# housing_percentage = df['housing'].value_counts(normalize=True)*100
# yes_housing_percentage = housing_percentage.get('yes',0)
# print(f"Proportion of clients that have housing loans is {yes_housing_percentage:.2f}%.")

# '''How many clients have personal loans?'''

# loan_counts = df['loan'].value_counts()
# print(loan_counts)

# loan_percentage = df['loan'].value_counts(normalize=True)*100
# yes_loan_percentage = loan_percentage.get('yes',0)
# print(f"Proportion of clients that have personal loans is {yes_loan_percentage:.2f}%.")

# '''What are the communication types used for contacting clients during the campaign?'''

# contact_counts = df['contact'].value_counts()
# print("Frequency count:", contact_counts)

# contact_percentage = df['contact'].value_counts(normalize=True) * 100
# print("\nRelative frequency (percentage):\n", contact_percentage)

# unique_contacts = df['contact'].nunique()
# print("\nNumber of unique contact modes:", unique_contacts)

# contact_mode = df['contact'].mode()[0]
# print("\nMost common contact mode (Mode):", contact_mode)

# #Pie chart for contact mode distribution
# plt.figure(figsize = (6,6))
# contact_counts.plot.pie(autopct= '%1.1f%%', startangle = 90, color = sns.color_palette("viridis"))
# plt.title("Contact Mode Distribution")
# plt.ylabel('')
# plt.show()

# '''What is the distribution of the last contact day of the month?'''

# day_count = df['day'].value_counts()
# print("Day count:", day_count)

# day_percentage = df['day'].value_counts(normalize=True)*100
# print("Day percentage:",day_percentage)

# '''How does the last contact month vary among the clients?'''

# month_frequency = df['month'].value_counts()
# print("Month frequency distribution:", month_frequency)

# month_percentage = df['month'].value_counts(normalize=True)*100
# print("Month percentage distribution:", month_percentage)

# month_map =  {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
# df['month_numeric'] = df['month'].map(month_map)

# std_deviation = df['month_numeric'].std()
# print(f"Standard deviation: {std_deviation:.2f}")

# variance = df['month_numeric'].var()
# print(f"Variance: {variance:.2f}")

# '''What is the distribution of the duration of the last contact?'''

# bins = [0, 50, 100, 200, 300, 500, 1000, 2000, float('inf')]
# labels = ['0-50', '51-100', '101-200', '201-300', '301-500', '501-1000', '1001-2000', '20001+']

# df['duration_bins'] = pd.cut(df['duration'], bins=bins, labels=labels, right=True)
# duration_frequency = df['duration_bins'].value_counts().sort_index()
# print("Frequency distribution:", duration_frequency)

# duration_percentage = df['duration_bins'].value_counts(normalize=True).sort_index()*100
# print("Percentage_distribution:", duration_percentage)

# '''How many contacts were performed during the campaign for each client?'''

# contact_frequency = df['campaign'].value_counts().sort_index()
# print("Number of contacts performed and Frequency:", contact_frequency)


# '''What is the distribution of the number of days passed since the client was last contacted from a previous campaign?'''

# bins = [-1,0,100,200,300,400,500,600,700,800,900, float('inf')]
# labels = ['Never Contacted', '1-100', '101-200', '201-300', '301-400', '401-500', '501-600', '601-700','701-800', '801-900', '900+']

# df['pdays_bins'] = pd.cut(df['pdays'], bins=bins, labels=labels, right=False)

# pdays_frequency = df['pdays_bins'].value_counts().sort_index()

# pdays_percentage = df['pdays_bins'].value_counts(normalize=True).sort_index()*100

# print("Frequency distribution (days passed for contacted clients):", pdays_frequency)
# print("Percentage distribution (days passed for contacted clients):", pdays_percentage)

# '''How many contacts were performed before the current campaign for each client?'''

# previous_counts = df['previous'].value_counts()
# previous_counts.index.name = None
# print("Number of contacts performed before the current campaign for each client:\n" + str(previous_counts))

# '''What were the outcomes of the previous marketing campaigns?'''

# outcome_counts = df['poutcome'].value_counts()
# print(outcome_counts)

# outcome_percentage = df['poutcome'].value_counts(normalize=True)*100
# print(outcome_percentage)

# '''What is the distribution of clients who subscribed to a term deposit vs. those who did not?'''

# y_frequency = df['y'].value_counts()
# print("Frequency Distribution:", y_frequency)

# y_percentage = (df['y'].value_counts(normalize=True)*100).round(2)
# print("Percentage Distribution:", y_percentage)

'''Are there any correlations between different attributes and the likelihood of subscribing to a term deposit?'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix


#Preprocessing
df.fillna({
    col: df[col].mode()[0] if df[col].dtype == 'object' else df[col].median()
    for col in df.columns
    },inplace=True)

#Encode the target variable 'y' (yes-> 1, no -> 0)
df['y_encoded'] = df['y'].apply(lambda x:1 if x== 'yes' else 0)

#One-hot encode categorical values
categorical_cols = df.select_dtypes(include=['object']).columns.drop('y')
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


#Prepare data for modeling
X = df_encoded.drop(columns=['y', 'y_encoded'])
y = df_encoded['y_encoded']

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#Fit into a logistic regression model
model = LogisticRegression(max_iter=500, solver='liblinear', random_state=42)
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:,1]

#Evaluate the model
print("Classification Report:", classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Compute and display ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.2f}")

#Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

#Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label = f"ROC Curve (AUC = {roc_auc:.2f})", color='blue')
plt.plot([0,1], [0,1], linestyle= '--', color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positie Rate')
plt.title('ROC Curve')
plt.legend()

#Display the coefficients of the model
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print("Top Features Influencing Subscription Likelihood:")
print(coefficients.head(10))
print("\nLeast Influential Features:")
print(coefficients.tail(10))

# Conclusions
print("\n### Conclusion: Correlations Between Attributes and Subscription Likelihood ###\n")

# Strong Positive Correlations
print("1. Strong Positive Correlations:")
print("   - Call Duration (duration):")
print("     Longer call durations significantly increase the likelihood of subscribing.")
print("   - Previous Campaign Success (poutcome_success):")
print("     Clients with successful outcomes in previous campaigns are more likely to subscribe.")
print("   - Months (e.g., March and October):")
print("     Subscriptions are more likely in specific months, possibly due to seasonal campaigns.\n")

# Negative Correlations
print("2. Negative Correlations:")
print("   - Unknown Previous Outcome (poutcome_unknown):")
print("     Clients with unknown past outcomes are less likely to subscribe.")
print("   - Housing Loans (housing_yes):")
print("     Clients with existing housing loans are less likely to subscribe due to financial commitments.")
print("   - Unknown Contact Method (contact_unknown):")
print("     Unknown contact methods are associated with lower subscription rates, highlighting the importance of reliable communication.\n")

# Insights from Logistic Regression Coefficients
print("3. Insights from Logistic Regression Coefficients:")
print("   - Top Positive Features:")
print("     * Duration, Previous Campaign Success, and Seasonal Timing (specific months).")
print("   - Top Negative Features:")
print("     * Unknown Previous Outcome, Housing Loan Status, and Unknown Contact Methods.\n")

# ROC-AUC Score Interpretation
print("4. Model Performance:")
print(f"   - The ROC-AUC Score is {roc_auc:.2f}.")
print("   - This indicates the model has a strong ability to distinguish between clients who subscribe and those who do not.\n")

# Summary
print("### Final Summary ###")
print("Yes, there are clear correlations between attributes and the likelihood of subscribing to a term deposit.")
print("The strongest drivers of subscription are longer call durations, successful previous campaigns, and seasonal timing.")
print("Factors like housing loans and unknown engagement history negatively affect subscription rates.")