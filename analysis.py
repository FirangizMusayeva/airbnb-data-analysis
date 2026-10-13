import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Örnek Airbnb veri kümesi oluşturma
np.random.seed(42)
n = 1000
data = {
    "accommodates": np.random.randint(1, 8, n),
    "bathrooms": np.random.choice([1, 2, 3], n),
    "is_entire_home": np.random.choice([0, 1], n),
}
df = pd.DataFrame(data)
df["price"] = (
    df["accommodates"] * 25
    + df["bathrooms"] * 20
    + df["is_entire_home"] * 50
    + np.random.normal(0, 10, n)
)

# 2. Model eğitimi ve değerlendirme
X = df[["accommodates", "bathrooms", "is_entire_home"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

print("Airbnb fiyat tahmin modeli basariyla egitildi!")
print("R2 Dogruluk Skoru:", round(model.score(X_test, y_test), 2))
