import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

data = pd.read_csv("england-premier-league-matches-2018-to-2019-stats (1).csv")

data['Match Outcome'] = np.where(data['home_team_goal_count'] > data['away_team_goal_count'], "Home Win", "Away Win")
data['Match Outcome'] = np.where(data['home_team_goal_count'] == data['away_team_goal_count'], "Draw", data['Match Outcome'])

data = data.drop(['timestamp', 'date_GMT', 'status', 'home_team_name','away_team_name', 'referee', 'stadium_name', 'home_team_goal_timings', 'away_team_goal_timings', 'home_team_goal_count', 'away_team_goal_count', 'home_team_goal_count_half_time', 'away_team_goal_count_half_time'], axis=1)
X = data.drop('Match Outcome', axis=1)
y = data['Match Outcome']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('pca', PCA()),                
    ('clf', LogisticRegression())  
])


param_grid = {
    'pca__n_components': [i for i in range(1,20)] 
}


grid_search = GridSearchCV(pipeline, param_grid, scoring='f1_macro')
grid_search.fit(X_train, y_train)


best_n_components = grid_search.best_params_['pca__n_components']
print("Best Number of Components:", best_n_components)


best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))