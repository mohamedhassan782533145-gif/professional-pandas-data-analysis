import pandas as pd

# ============================================================

df = pd.read_csv("diamonds.csv")
# ===========================================================

print("\n========== DATA OVERVIEW ==========")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nIndex:")
print(df.index)

print("\nData Types:")
print(df.dtypes)

print("\nFirst Rows:")
print(df.head())

print("\nRandom Sample:")
print(df.sample(5))


# ============================================================

print("\n========== DATA QUALITY ==========")

print("\nMissing Values:")
print(df.isna().sum())

print("\nDuplicated Rows:")
print(df.duplicated().sum())

print("\nUnique Values:")
print(df.nunique())


# ============================================================

print("\n========== DATA CLEANING ==========")

df = df.drop_duplicates()

df["cut"] = df["cut"].replace({
    "Very Good": "VeryGood"
})

df["cut"] = df["cut"].astype("str")

df["color"] = df["color"].astype("str")

df["clarity"] = df["clarity"].astype("str")

df["price"] = df["price"].astype("float64")


# ============================================================


print("\n========== CATEGORICAL ANALYSIS ==========")

print("\nCut:")
print(df["cut"].value_counts())

print("\nColor:")
print(df["color"].value_counts())

print("\nClarity:")
print(df["clarity"].value_counts())

print("\nUnique Cuts:")
print(df["cut"].unique())

print("\nNumber of Cuts:")
print(df["cut"].nunique())


# ============================================================


print("\n========== FILTERING ==========")

premium_diamonds = df.loc[
    df["cut"] == "Premium"
]

print("\nPremium Diamonds:")
print(premium_diamonds.head())

large_diamonds = df.loc[
    df["carat"].between(1, 2)
]

print("\nDiamonds Between 1 and 2 Carat:")
print(large_diamonds.head())

expensive_diamonds = df.query(
    "price > 3000"
)

print("\nExpensive Diamonds:")
print(expensive_diamonds.head())

selected_diamonds = df[
    df["color"].isin(["D", "E", "F"])
]

print("\nTop Color Categories:")
print(selected_diamonds.head())


# ===========================================================

print("\n========== SORTING ==========")

highest_price = df.sort_values(
    "price",
    ascending=False
)

print("\nMost Expensive Diamonds:")
print(
    highest_price[
        [
            "carat",
            "cut",
            "color",
            "clarity",
            "price"
        ]
    ].head(10)
)


highest_carat = df.sort_values(
    "carat",
    ascending=False
)

print("\nHighest Carat:")
print(
    highest_carat[
        [
            "carat",
            "cut",
            "price"
        ]
    ].head(10)
)


# ============================================================


print("\n========== FEATURE ENGINEERING ==========")

df["price_per_carat"] = (
    df["price"] / df["carat"]
)

df["volume"] = (
    df["x"] *
    df["y"] *
    df["z"]
)

df["value_category"] = df["price"].where(
    df["price"] < 2000,
    "High Value"
)

df["carat_level"] = df["carat"].clip(
    upper=3
)


# ============================================================


print("\n========== CUT RANKING ==========")

cut_ranking = {
    "Ideal": 5,
    "Premium": 4,
    "VeryGood": 3,
    "Good": 2,
    "Fair": 1
}

df["cut_score"] = df["cut"].map(
    cut_ranking
)

print(
    df[
        [
            "cut",
            "cut_score"
        ]
    ].head()
)


# ============================================================


print("\n========== GROUP ANALYSIS ==========")

cut_analysis = (
    df.groupby("cut")
      .agg(
          Average_Price=("price", "mean"),
          Median_Price=("price", "median"),
          Maximum_Price=("price", "max"),
          Minimum_Price=("price", "min"),
          Average_Carat=("carat", "mean"),
          Total_Diamonds=("price", "count")
      )
      .sort_values(
          "Average_Price",
          ascending=False
      )
)

print(cut_analysis)


# ===========================================================

color_analysis = (
    df.groupby("color")
      .agg(
          Average_Price=("price", "mean"),
          Average_Carat=("carat", "mean"),
          Total_Diamonds=("price", "count")
      )
      .sort_values(
          "Average_Price",
          ascending=False
      )
)

print("\nColor Analysis:")
print(color_analysis)


# ============================================================

clarity_analysis = (
    df.groupby("clarity")
      .agg(
          Average_Price=("price", "mean"),
          Average_Carat=("carat", "mean"),
          Maximum_Price=("price", "max"),
          Total_Diamonds=("price", "count")
      )
      .sort_values(
          "Average_Price",
          ascending=False
      )
)

print("\nClarity Analysis:")
print(clarity_analysis)


# ============================================================


print("\n========== MULTI-DIMENSIONAL ANALYSIS ==========")

cut_color_analysis = (
    df.groupby(
        ["cut", "color"]
    )
    .agg(
        Average_Price=("price", "mean"),
        Average_Carat=("carat", "mean"),
        Diamond_Count=("price", "count")
    )
    .sort_values(
        "Average_Price",
        ascending=False
    )
)

print(cut_color_analysis.head(15))


# ============================================================

df["cut_average_price"] = (
    df.groupby("cut")["price"]
      .transform("mean")
)

df["price_vs_cut_average"] = (
    df["price"] -
    df["cut_average_price"]
)


# ============================================================

df["price_level"] = df["price"].apply(
    lambda x:
        "Low"
        if x < 1000
        else
        "Medium"
        if x < 3000
        else
        "High"
)


# ============================================================


df["expensive_price"] = df["price"].where(
    df["price"] >= 3000,
    0
)

df["discount_flag"] = df["price"].mask(
    df["price"] < 1000,
    0
)


# ============================================================


print("\n========== PIVOT TABLE ==========")

pivot_analysis = pd.pivot_table(
    df,
    values="price",
    index="cut",
    columns="color",
    aggfunc="mean"
)

print(pivot_analysis)


# ============================================================


df["price_rank"] = (
    df["price"]
    .rank(
        ascending=False,
        method="dense"
    )
)

print("\nTop Ranked Diamonds:")

print(
    df.sort_values(
        "price_rank"
    )[
        [
            "carat",
            "cut",
            "color",
            "clarity",
            "price",
            "price_rank"
        ]
    ].head(10)
)


# ============================================================

high_value = df.query(
    "price >= 5000 and carat >= 1"
)

premium_segment = df.query(
    "cut == 'Premium' and price >= 3000"
)

quality_segment = df[
    df["clarity"].isin(
        ["IF", "VVS1", "VVS2"]
    )
]

print("\nHigh Value Segment:")
print(high_value.head())

print("\nPremium Segment:")
print(premium_segment.head())

print("\nHigh Clarity Segment:")
print(quality_segment.head())


# ============================================================


print("\n========== BUSINESS INSIGHTS ==========")

best_cut = (
    cut_analysis["Average_Price"]
    .idxmax()
)

best_color = (
    color_analysis["Average_Price"]
    .idxmax()
)

best_clarity = (
    clarity_analysis["Average_Price"]
    .idxmax()
)

most_expensive = df.loc[
    df["price"].idxmax()
]

print(
    "Highest Average Price Cut:",
    best_cut
)

print(
    "Highest Average Price Color:",
    best_color
)

print(
    "Highest Average Price Clarity:",
    best_clarity
)

print(
    "\nMost Expensive Diamond:"
)

print(most_expensive)


# ============================================================


print("\n========== MEMORY USAGE ==========")

print(
    df.memory_usage(
        deep=True
    )
)

print(
    "\nTotal Memory:",
    df.memory_usage(
        deep=True
    ).sum()
)


# ============================================================


final_columns = [
    "carat",
    "cut",
    "color",
    "clarity",
    "depth",
    "table",
    "price",
    "price_per_carat",
    "volume",
    "cut_score",
    "price_level",
    "price_rank"
]

final_df = df.loc[
    :,
    final_columns
]


# ============================================================

final_df.to_csv(
    "diamonds_final_analysis.csv",
    index=False
)

cut_analysis.to_csv(
    "cut_analysis.csv"
)

color_analysis.to_csv(
    "color_analysis.csv"
)

clarity_analysis.to_csv(
    "clarity_analysis.csv"
)

pivot_analysis.to_csv(
    "price_pivot_analysis.csv"
)


print(
    "\n========================================"
)

print(
    "Professional Pandas Analysis Completed!"
)

print(
    "Final dataset exported successfully."
)

print(
    "========================================"
)