# from data_download import dst
import os
import pandas as pd
from pathlib import Path


df = {}
for csv_file in Path("/Users/busegungor/model1/data").glob("*.csv"):
    name = csv_file.stem            # filename without .csv extension
    df[name] = pd.read_csv(csv_file)


products = df["products"].copy()
categories = df["categories"].copy()
regions = df["regions"].copy()
returns = df["returns"].copy()
orders = df["orders"].copy()
order_items = df["order_items"].copy()
customers = df["customers"].copy()


# ============================================================
# Merge — start from order_items (the most detailed table) and enrich with everything else
# ============================================================
main = (order_items
    .merge(orders,     on="order_id",      how="left")
    .merge(regions,    on="region_id",     how="left")
    .merge(products,   on="product_id",    how="left")
    .merge(categories, on="category_id",   how="left")
    .merge(returns,    on="order_item_id", how="left")
)

# main.to_csv("/Users/busegungor/model1/data/master_data.csv")








