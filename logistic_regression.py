import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
# import the data and add in the 'Match Outcome' feature
data = pd.read_csv("england-premier-league-matches-2018-to-2019-stats (1).csv")

data['Match Outcome'] = np.where(data['home_team_goal_count'] > data['away_team_goal_count'], "Home Win", "Away Win")
data['Match Outcome'] = np.where(data['home_team_goal_count'] == data['away_team_goal_count'], "Draw", data['Match Outcome'])

X = data[['Pre-Match PPG (Home)', 'Pre-Match PPG (Away)', 'team_a_xg', 'team_b_xg', 'home_ppg', 'away_ppg']]
y = data['Match Outcome']
print(np.unique(y))
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Initialize logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)


# Train the model
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)
output = data.tail(int(len(data)*0.2))
output['Outcome Prediction'] = y_pred

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))
# output.to_csv('output.csv')