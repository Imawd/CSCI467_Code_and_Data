import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight
    
data = pd.read_csv("england-premier-league-matches-2018-to-2019-stats (1).csv")

data['Match Outcome'] = np.where(data['home_team_goal_count'] > data['away_team_goal_count'], "Home Win", "Away Win")
data['Match Outcome'] = np.where(data['home_team_goal_count'] == data['away_team_goal_count'], "Draw", data['Match Outcome'])
data = data.drop(['timestamp', 'date_GMT', 'status', 'home_team_name','away_team_name', 'referee', 'stadium_name', 'home_team_goal_timings', 'away_team_goal_timings', 'home_team_goal_count', 'away_team_goal_count', 'home_team_goal_count_half_time', 'away_team_goal_count_half_time'], axis=1)
X = data.drop('Match Outcome', axis=1)
y = data['Match Outcome']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


print("Label Mapping:")
for label, encoded_label in label_mapping.items():
    print(f"{label}: {encoded_label}")


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=3)


hidden_layers = [1, 2]
neurons = [4, 8, 16, 32, 64]
dropout_rates = [0.1, 0.2, 0.3, 0.4]

best_accuracy = 0
best_model = None
best_params = {}

for num_layers in hidden_layers:
    for num_neurons in neurons:
        for dropout_rate in dropout_rates:
            print(f"Training model with {num_layers} hidden layer(s), {num_neurons} neurons, and dropout rate {dropout_rate}...")

            # Define the model architecture
            model = Sequential()
            model.add(Dense(num_neurons, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dropout(dropout_rate))
            for _ in range(num_layers - 1):
                model.add(Dense(num_neurons, activation='relu'))
                model.add(Dropout(dropout_rate))
            model.add(Dense(3, activation='softmax'))


            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    
            model.fit(X_train, y_train_encoded, epochs=20, batch_size=32, verbose=0)

         
            _, accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)

          
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_params = {'hidden_layers': num_layers, 'neurons': num_neurons, 'dropout_rate': dropout_rate}

print("Best model accuracy:", best_accuracy)
print("Best model parameters:", best_params)


y_pred_probs = best_model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)


class_names = ['Home Win', 'Away Win', 'Draw']
y_test_labels = [class_names[label] for label in y_test]
y_pred_labels = [class_names[label] for label in y_pred_labels]


report = classification_report(y_test_labels, y_pred_labels, target_names=class_names)

print("Classification Report:")
print(report)