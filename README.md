# F1 Race Winner Predictor

A machine learning project that predicts the winner of the next Formula 1 race
using historical race results, qualifying data, and recent driver/team form.

## What it does
- Merges race, driver, constructor, and qualifying data into one dataset
- Engineers rolling-average "form" features (recent finishing position, recent
  points, team strength) without leaking future race data into predictions
- Trains a Random Forest classifier to estimate each driver's win probability
- Visualizes win probability by driver in a bar chart
- Prints the predicted winner

## Results
- ROC-AUC: 0.945 — strong at ranking drivers by likelihood of winning
- Recall (win class): 0.82 — correctly flags most actual winners
- Model sensibly favors drivers with strong grid position and recent form
  (e.g. correctly favored Norris given pole position and season form)

## Tech stack
Python, pandas, scikit-learn, matplotlib, seaborn

## Dataset
Historical F1 data (races, results, drivers, constructors, qualifying) —
[https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020]

## How to run
\`\`\`
pip install -r requirements.txt
python main.py
\`\`\`

## Author
Jenisha Varde — [github.com/Jenisha006](https://github.com/Jenisha006)