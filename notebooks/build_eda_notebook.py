"""Generate notebooks/eda.ipynb - the Task 3 EDA / ad-hoc analytics notebook.
Run once:  python notebooks/build_eda_notebook.py
Then open eda.ipynb in Jupyter, or execute headless to HTML:
    jupyter nbconvert --to html --execute notebooks/eda.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
def md(t): c.append(nbf.v4.new_markdown_cell(t))
def code(t): c.append(nbf.v4.new_code_cell(t))

md("""# CIFAR-10 Pipeline - Exploratory Data Analysis (Task 3)

This notebook is the **ad-hoc analytics / reporting** consumer of the pipeline.
It connects to the PostgreSQL analytics store and explores the **ingested**
image-feature data (the `images` table populated by the ETL in Task 2).

It covers: levels of measurement, missing values & outliers, variable
distributions, relationships between features and with the target, and a
comparison of the ingested/stored data against the original source.""")

code("""import os
from urllib.parse import quote_plus
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))            # find repo .env from anywhere
U, P = os.getenv("POSTGRES_USER","postgres"), os.getenv("POSTGRES_PASSWORD","")
H, PORT, DB = os.getenv("POSTGRES_HOST","localhost"), os.getenv("POSTGRES_PORT","5432"), os.getenv("POSTGRES_DB","mlops_db")
auth = quote_plus(U) if not P else f"{quote_plus(U)}:{quote_plus(P)}"
engine = create_engine(f"postgresql+psycopg2://{auth}@{H}:{PORT}/{DB}")

os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid")
NUM = ["brightness","contrast","mean_r","mean_g","mean_b","std_r","std_g","std_b"]

df = pd.read_sql("SELECT * FROM images", engine)
print("Loaded", df.shape[0], "rows,", df.shape[1], "columns")
df.head()""")

md("""## 1. Data overview & levels of measurement

| Variable | Statistical type | Level of measurement |
|---|---|---|
| `label` | Categorical (target) | Nominal |
| `class_name` | Categorical (target) | Nominal |
| `split` | Categorical | Nominal |
| `mean_r/g/b`, `std_r/g/b`, `brightness`, `contrast` | Continuous (features) | Ratio (true zero) |
| `image_id` | Identifier | — (key, not analysed) |""")

code("""print(df.dtypes)
df[NUM].describe().T.round(2)""")

md("## 2. Missing values & outliers")

code("""print("Missing values per column:")
print(df.isnull().sum())

print("\\nIQR outliers per feature:")
for f in NUM:
    q1, q3 = df[f].quantile([.25, .75]); iqr = q3 - q1
    mask = (df[f] < q1 - 1.5*iqr) | (df[f] > q3 + 1.5*iqr)
    print(f"  {f:11s}: {int(mask.sum()):5d}  ({100*mask.mean():.1f}%)")""")

code("""fig, ax = plt.subplots(figsize=(10,4))
df[NUM].plot(kind="box", ax=ax); ax.set_title("Feature spread & outliers")
plt.xticks(rotation=30); plt.tight_layout(); plt.savefig("figures/boxplots.png"); plt.show()""")

md("## 3. Distributions")

code("""fig, ax = plt.subplots(figsize=(9,4))
df[df.split=="train"]["class_name"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_title("Class balance (train)"); ax.set_ylabel("count")
plt.tight_layout(); plt.savefig("figures/class_balance.png"); plt.show()""")

code("""fig, axes = plt.subplots(2, 4, figsize=(15,7))
for f, ax in zip(NUM, axes.ravel()):
    df[f].plot(kind="hist", bins=40, ax=ax, title=f)
plt.tight_layout(); plt.savefig("figures/feature_hists.png"); plt.show()""")

md("## 4. Relationships (feature-feature and feature-target)")

code("""fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(df[NUM].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature correlation matrix")
plt.tight_layout(); plt.savefig("figures/correlation.png"); plt.show()""")

code("""fig, axes = plt.subplots(1, 2, figsize=(16,5))
sns.boxplot(data=df[df.split=="train"], x="class_name", y="brightness", ax=axes[0])
sns.boxplot(data=df[df.split=="train"], x="class_name", y="contrast", ax=axes[1])
for a in axes: a.tick_params(axis="x", rotation=45)
axes[0].set_title("Brightness by class"); axes[1].set_title("Contrast by class")
plt.tight_layout(); plt.savefig("figures/feature_vs_target.png"); plt.show()""")

md("""## 5. Ingested/stored data vs original source

The ingested data is a faithful copy of the torchvision CIFAR-10 source, but its
*representation* differs: the pipeline stores derived **feature columns** in
PostgreSQL (relational) and the raw **pixel arrays** in Redis (key-value),
rather than the original single compressed tarball. Below we compare the train
vs test splits to confirm they are drawn from the same distribution.""")

code("""comp = df.groupby("split")[NUM].mean().T.round(2)
comp["abs_diff"] = (comp.get("train",0) - comp.get("test",0)).abs().round(2)
print("Per-feature mean by split:")
comp""")

md("""## Summary

*Fill in for the report:* class balance (is the target uniform?), which features
show spread/outliers, notable feature-feature correlations, whether any feature
separates the classes (feature-vs-target signal), and confirmation that train
and test share the same distribution.""")

nb["cells"] = c
with open("notebooks/eda.ipynb", "w") as f:
    nbf.write(nb, f)
print("Wrote notebooks/eda.ipynb with", len(c), "cells")
