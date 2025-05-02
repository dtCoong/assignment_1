import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import requests
import os

url = 'https://www.footballtransfers.com/en/values/actions/most-valuable-football-players/overview'

headers = {
    'user-agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0',
    'x-requested-with':'XMLHttpRequest',
    'referer': 'https://www.footballtransfers.com/en/values/players/most-valuable-players/playing-in-uk-premier-league',
    'content-type': 'application/x-www-form-urlencoded; charset=UTF-8'
}

rows = []
def get_data(i):
    payload = {
        'orderBy': 'estimated_value',
        'orderByDescending': 1,
        'page': i,
        'pages': 0,
        'pageItems': 25,
        'positionGroupId': all,
        'mainPositionId': all,
        'playerRoleId': all,
        'age': all,
        'countryId': all,
        'tournamentId': 31
    }
    x = requests.post(url,data=payload, headers=headers)
    data = x.json()
    players = data.get("records", [])
    for i in players:
        rows.append(i) 

for i in range(1,23):
    get_data(i)
    
    
df1 = pd.DataFrame(rows)
df1 = df1[['player_name','estimated_value']]
df1.rename(columns={ "player_name": "player"}, inplace=True)

df2 = pd.read_csv('results.csv')
df2 = df2[df2['minutes'] >= 900]
df2 = df2[['player','minutes']]

df = df2.merge(df1, how='left', on=['player'])
df.dropna(inplace = True)
print(df)
df.to_csv("player_transfer_value.csv",index=False,encoding='utf-8-sig')

def convert_market_value(value):
    if isinstance(value, str):
        value = value.replace('€', '').strip()
        if 'M' in value:
            return float(value.replace('M', '').strip()) * 1e6
        elif 'B' in value:
            return float(value.replace('B', '').strip()) * 1e9
        else:
            try:
                return float(value)
            except ValueError:
                return np.nan 
    return value

stats_df = pd.read_csv('results.csv')
values_df = pd.read_csv('player_transfer_value.csv')

df_filtered = stats_df.merge(values_df,how = 'left', on=['player','minutes'])
df_filtered = df_filtered[df_filtered['minutes'] >= 900]
df_filtered.dropna(inplace=True)
df_filtered.replace('N/a', np.nan, inplace=True)
numerical_features = [col for col in df_filtered.columns][5:]

df_filtered['estimated_value'] = df_filtered['estimated_value'].apply(convert_market_value)
df_filtered.fillna(0, inplace=True)


models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=0),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

# Chia dữ liệu
X = df_filtered[numerical_features].drop(columns=['estimated_value'])
y = df_filtered['estimated_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Kết quả đánh giá
results = []

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE (€)': round(mae),
        'RMSE (€)': round(rmse),
        'R²': round(r2, 4)
    })

# In bảng kết quả
results_df = pd.DataFrame(results)
print(results_df)

# Thiết lập figure
plt.figure(figsize=(14, 4))

# Biểu đồ MAE
plt.subplot(1, 3, 1)
plt.bar(results_df['Model'], results_df['MAE (€)'], color='skyblue')
plt.title('MAE (Million €)')
plt.ylabel('MAE (€)')
plt.xticks(rotation=45)

# Biểu đồ RMSE
plt.subplot(1, 3, 2)
plt.bar(results_df['Model'], results_df['RMSE (€)'], color='orange')
plt.title('RMSE (Million €)')
plt.ylabel('RMSE (€)')
plt.xticks(rotation=45)

# Biểu đồ R²
plt.subplot(1, 3, 3)
plt.bar(results_df['Model'], results_df['R²'], color='lightgreen')
plt.title('R² Score')
plt.ylabel('R²')
plt.xticks(rotation=45)

# Hiển thị
os.makedirs("image/bai4", exist_ok=True)
plt.tight_layout()
plt.savefig(f'image/bai4/sosanhmohinh.png',dpi = 300)

#RandomForest
X = df_filtered[numerical_features].drop(columns=['estimated_value'])
y = df_filtered['estimated_value']
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)
importances = model.feature_importances_
feature_names = X.columns

# Tạo DataFrame sắp xếp theo độ quan trọng
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top_features = feature_importance_df['Feature'].head(15).tolist()
print("Top features:", top_features)

# Huấn luyện lại mô hình với 15 đặc trưng 
X_top15 = df_filtered[top_features]
y = df_filtered['estimated_value']
X_train, X_test, y_train, y_test = train_test_split(X_top15, y, test_size=0.2, random_state=0)

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:,.0f} €")
print(f"RMSE: {rmse:,.0f} €")
print(f"R²: {r2:.4f}")

df_filtered['predicted_value'] = model.predict(df_filtered[top_features])
df_filtered['gap (€)'] = df_filtered['predicted_value'] - df_filtered['estimated_value']
df_filtered['abs_gap (€)'] = df_filtered['gap (€)'].abs()
df_filtered = df_filtered[['player','predicted_value','estimated_value','gap (€)']]
df_filtered.to_csv('player_value_predictions.csv', index=False, encoding='utf-8-sig')