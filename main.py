import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

sns.set_style("darkgrid")

races = pd.read_csv("races.csv")
results = pd.read_csv("results.csv")
drivers = pd.read_csv("drivers.csv")
constructors = pd.read_csv("constructors.csv")
qualifying = pd.read_csv("qualifying.csv")

df = results.merge(races[['raceId', 'year', 'round', 'circuitId']], on='raceId')
df = df.merge(drivers[['driverId', 'driverRef']], on='driverId')
df = df.merge(constructors[['constructorId', 'name']], on='constructorId', suffixes=('', '_team'))

quali = qualifying[['raceId', 'driverId', 'position']].rename(columns={'position': 'quali_position'})
df = df.merge(quali, on=['raceId', 'driverId'], how='left')

print(df.shape)
df.head()

df['grid'] = pd.to_numeric(df['grid'], errors='coerce')
df['positionOrder'] = pd.to_numeric(df['positionOrder'], errors='coerce')
df['quali_position'] = pd.to_numeric(df['quali_position'], errors='coerce')

df['won'] = (df['positionOrder'] == 1).astype(int)

df = df.dropna(subset=['grid', 'positionOrder'])

df = df.sort_values(['driverId', 'year', 'round'])

df['recent_form'] = df.groupby('driverId')['positionOrder'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)

df['recent_points'] = df.groupby('driverId')['points'].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)

df['team_form'] = df.groupby(['name', 'year'])['positionOrder'].transform(
    lambda x: x.shift(1).expanding().mean()
)

df['recent_form'] = df['recent_form'].fillna(df['positionOrder'].mean())
df['recent_points'] = df['recent_points'].fillna(0)
df['team_form'] = df['team_form'].fillna(df['positionOrder'].mean())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

win_by_grid = df.groupby('grid')['won'].mean().reset_index()
axes[0].bar(win_by_grid['grid'][:10], win_by_grid['won'][:10], color='crimson')
axes[0].set_title("Win Rate by Starting Grid Position")
axes[0].set_xlabel("Grid Position")
axes[0].set_ylabel("Win Rate")

top_teams = df[df['won']==1]['name'].value_counts().head(10)
axes[1].barh(top_teams.index[::-1], top_teams.values[::-1], color='steelblue')
axes[1].set_title("Most Race Wins by Constructor")

plt.tight_layout()
plt.show()

features = ['grid', 'quali_position', 'recent_form', 'recent_points', 'team_form']
df_model = df.dropna(subset=features)

X = df_model[features]
y = df_model['won']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=300, max_depth=6, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

print("Test accuracy:", model.score(X_test, y_test))

latest_idx = df.groupby('driverId')['raceId'].transform('max') == df['raceId']
latest_per_driver = df[latest_idx].copy()

current_season = df['year'].max()
next_race_df = latest_per_driver[latest_per_driver['year'] == current_season].copy()

next_race_df['grid'] = next_race_df.groupby('driverId')['grid'].transform('mean')
next_race_df['quali_position'] = next_race_df['grid']

print(next_race_df[['driverRef', 'grid', 'recent_form', 'recent_points', 'team_form']])

probs = model.predict_proba(next_race_df[features])[:, 1]
next_race_df['win_probability'] = probs

ranked = next_race_df.sort_values('win_probability', ascending=False)

plt.figure(figsize=(8,5))
plt.barh(ranked['driverRef'][:10][::-1], ranked['win_probability'][:10][::-1], color='gold')
plt.xlabel("Predicted Win Probability")
plt.title("Next Race Win Prediction")
plt.tight_layout()
plt.show()

predicted_winner = ranked.iloc[0]['driverRef']
print(f"\n\033[1mPREDICTED WINNER: {predicted_winner.upper()}\033[0m\n")

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))