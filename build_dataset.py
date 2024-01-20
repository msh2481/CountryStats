import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from itertools import chain, cycle, islice, product
from typing import Annotated, Any

import einops as ein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from beartype import beartype as typed
from beartype.door import die_if_unbearable as assert_type
from beartype.typing import Callable, Iterable
from beartype.vale import Is
from jaxtyping import Bool, Float, Int
from numpy import ndarray as ND
from tqdm import tqdm

Country = str
Metric = str
Year = int

hdi_df = pd.read_csv("HDR21-22_Composite_indices_complete_time_series.csv")
pop_df = pd.read_csv("world_population.csv")

pop_df["1990 Population"] = pop_df["2000 Population"] - (
    pop_df["2010 Population"] - pop_df["2000 Population"]
)
years_with_populations = [1990, 2000, 2010, 2020, 2022]
for year in range(1990, 2021 + 1):
    if year not in years_with_populations:
        l_y = max([y for y in years_with_populations if y <= year])
        r_y = min([y for y in years_with_populations if y >= year])
        l_p = pop_df[f"{l_y} Population"]
        r_p = pop_df[f"{r_y} Population"]
        pop_df[f"{year} Population"] = (year - l_y) / (r_y - l_y) * (r_p - l_p) + l_p
    pop_df[f"{year} Population"] = pop_df[f"{year} Population"].astype(int)


metrics: list[str] = []
for col in hdi_df.columns:
    if (
        col in ["iso3", "country", "hdicode", "region"]
        or "rank" in col
        or "gdi_group" in col
        or "ihdi" in col
        or "ineq" in col
        or "loss" in col
    ):
        continue
    parts = list(col.split("_"))
    metric = "_".join(parts[:-1])
    metrics.append(metric)

data_list: list[dict[str, Any]] = []
used_iso3 = set()

for _, row in hdi_df.iterrows():
    iso3 = row["iso3"]
    if "." in iso3:
        continue
    if (pop_df["CCA3"] == iso3).sum() == 0:
        continue
    pop_row = pop_df[pop_df["CCA3"] == iso3].iloc[0]
    if iso3 not in used_iso3 and row["country"] != pop_row["Country/Territory"]:
        print("Matched", iso3, row["country"], pop_row["Country/Territory"])
    country = pop_row["Country/Territory"]
    used_iso3.add(iso3)

    for year in range(1990, 2021 + 1):
        to_append = {
            "country": country,
            "iso3": iso3,
            "year": year,
            "population": pop_row[f"{year} Population"],
        }
        for metric in metrics:
            to_append[metric] = row[f"{metric}_{year}"]
        data_list.append(to_append)


data = pd.DataFrame(data_list)
data.to_csv("data.csv")
