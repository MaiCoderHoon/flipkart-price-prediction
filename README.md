# 🛒 Flipkart E-Commerce - EDA & Price Prediction

![Dashboard](assets/dashboard.png)

A complete ML pipeline built on a Flipkart product dataset (~80,000 rows, 25 features) - covering EDA, feature engineering, leakage detection, model training, and evaluation end to end.

---

## 📌 The Problem

If we have a Flipkart product listing with details like category, brand, ratings, discount, seller info, and logistics - **can we accurately guess its final sale price?** Let's find out!

---

## 📊 Our Dataset

| Property | Value |
|---|---|
| Rows | 80,000 |
| Features | 25 |
| Target | `final_price` (₹) |
| Categories | Electronics, Fashion, Mobiles, Beauty, Toys, Sports, Appliances, Home & Kitchen |

We'll be looking at key columns like: `price`, `discount_percent`, `final_price`, `rating`, `review_count`, `units_sold`, `category`, `brand`, `seller`, `seller_city`, `product_score`, `delivery_days`, `warranty_months`, and a few others.

---

## 🗂️ How the Project is Organised

```
flipkart-price-prediction/
├── assets/
│   └── flipkard.csv       # The dataset we're working with
    └── dashboard.png
├── src/
│   └── pipeline.py                 # The full ML pipeline script
├── outputs/
│   ├── dashboard.png               # Visualizations for EDA and model performance
│   └── best_model.pkl              # Our best model, saved and ready to go
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ The ML Pipeline in Action

### 1. Feature Engineering
We've cooked up seven domain-specific features from the raw columns to give our model an edge:

| Feature | Description |
|---|---|
| `engagement` | `rating × log(1 + review_count)` - a mix of quality and popularity |
| `stock_sold_ratio` | `units_sold / (stock + units_sold)` - tells us how much demand there is |
| `listing_age_days` | Days since listing - how fresh the product is |
| `listing_year/month` | Breaking down the timeline |
| `payment_modes_count` | How many ways you can pay |

### 2. Spotting Data Leaks ⚠️
Here's a crucial thing we found during feature selection: `final_price = price × (1 - discount_percent / 100)`. That's a perfect, mathematical relationship! If we included `price` or `discount_amount` as features, it would be considered **data leakage**, artificially boosting our model's performance to a perfect score. To keep things honest, we excluded both of those features from our final model.

### 3. The Models We Trained

| Model | R² | MAE |
|---|---|---|
| Ridge Regression | ~0.11 | ~₹11,700 |
| Random Forest | ~0.11 | ~₹11,700 |
| Gradient Boosting | ~0.11 | ~₹11,700 |

> **A quick note on those scores:** Since we're using a synthetic dataset, the prices are uniformly distributed across all categories and have almost zero natural correlation to the other features. The pipeline itself is built properly; the low R² just reflects the nature of the dataset rather than a failure of the model. We made a conscious choice to catch the leakage and report honest scores.

### 4. Getting the Data Ready
- We used `StandardScaler` to handle our numerical features.
- We used `OneHotEncoder` (with handle_unknown='ignore') for categorical features.
- All of this is neatly bundled using `sklearn.compose.ColumnTransformer`.

---

## 📈 Some Cool Findings from our EDA

- We have **80,000 products** spanning 8 categories, listed between 2018 and 2023.
- The **top category for revenue** is Electronics.
- The **average discount** hovers around 21% across all categories.
- About **80% of the products can be returned**; interestingly, there's no major price difference compared to non-returnable items.
- The **Engagement score** we created (`rating × log(reviews)`) turned out to be our strongest predictor!

---

## 🚀 Want to try it out?

```bash
# 1. Grab a copy of the repo
git clone https://github.com/MaiCoderHoon/flipkart-price-prediction.git
cd flipkart-price-prediction

# 2. Install what you need
pip install -r requirements.txt

# 3. Let it rip!
python src/pipeline.py
```

Once it runs, you'll find the results in the `outputs/` folder:
- `dashboard.png` - A nice 9-panel figure showing our EDA and model evaluation.
- `best_model.pkl` - The best model we found, ready to make predictions.

---

## 🧠 What We Learned

- **Data leakage** is sneaky and can artificially inflate how good a model looks - always double-check how features relate to your target before training!
- **Synthetic datasets** often give you uniform targets, but the real value is in learning to build a pipeline that's robust and reproducible.
- **Feature engineering** can often make a bigger difference than which model you choose, especially when initial correlations are weak.
- Using **ColumnTransformer and Pipeline** keeps your preprocessing clean and ensures you don't contaminate your training data with test data.

---

## 🛠️ The Tech We Used

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `Matplotlib` · `Seaborn` · `joblib`
