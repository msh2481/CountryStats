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


@typed
def build_base() -> pd.DataFrame:
    hdi_df = pd.read_csv("HDR21-22_Composite_indices_complete_time_series.csv")
    pop_df = pd.read_csv("world_population.csv")
    gdp_df = pd.read_csv("GDP.csv")
    terror_df = pd.read_csv("terrorist-attacks.csv")
    w23_df = pd.read_csv("world-data-2023.csv")

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
            pop_df[f"{year} Population"] = (year - l_y) / (r_y - l_y) * (
                r_p - l_p
            ) + l_p
        pop_df[f"{year} Population"] = pop_df[f"{year} Population"].astype(int)

    hdi_metrics: list[str] = []
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
        hdi_metric = "_".join(parts[:-1])
        hdi_metrics.append(hdi_metric)

    data_list: list[dict[str, Any]] = []
    used_iso3 = set()

    for _, row in hdi_df.iterrows():
        iso3 = row["iso3"]
        if "." in iso3:
            continue
        if (pop_df["CCA3"] == iso3).sum() == 0:
            print(iso3, "not matched in pop_df")
            continue

        pop_row = pop_df[pop_df["CCA3"] == iso3].iloc[0]
        gdp_row = gdp_df[gdp_df["Country Code"] == iso3].iloc[0]

        if iso3 not in used_iso3 and row["country"] != pop_row["Country/Territory"]:
            print("Matched", iso3, row["country"], pop_row["Country/Territory"])

        country = pop_row["Country/Territory"]
        w23_row = None
        for _, w23_row_iter in w23_df.iterrows():
            if (
                w23_row_iter["Country"] == country
                or w23_row_iter["Country"] == pop_row["Country/Territory"]
            ):
                w23_row = w23_row_iter
                break
        used_iso3.add(iso3)

        for year in range(1990, 2021 + 1):
            terror_subset = terror_df[
                (terror_df["Code"] == iso3) & (terror_df["Year"] == year)
            ]
            if len(terror_subset) == 0:
                terrorism_deaths = None
            else:
                terrorism_deaths = terror_subset.iloc[0]["Terrorism deaths"]
            to_append = {
                "country": country,
                "iso3": iso3,
                "year": year,
                "population": pop_row[f"{year} Population"],
                "gdp": gdp_row[f"{year}"],
                "terrorism_deaths": terrorism_deaths,
                "army": w23_row["Armed Forces size"] if w23_row is not None else None,
                "area": w23_row["Land Area(Km2)"] if w23_row is not None else None,
                "birth_rate": w23_row["Birth Rate"] if w23_row is not None else None,
            }
            for hdi_metric in hdi_metrics:
                to_append[hdi_metric] = row[f"{hdi_metric}_{year}"]
            data_list.append(to_append)

    return pd.DataFrame(data_list)


@typed
def build_ideals() -> pd.DataFrame:
    ideal_df = pd.read_csv("IdealpointestimatesAll_Jul2023.csv")
    result = pd.DataFrame()
    result["iso3"] = ideal_df["iso3c"]
    result["year"] = ideal_df["session"] + 1945
    result["ideal"] = ideal_df["IdealPointAll"]
    return result


# data_base = build_base()
# data_base.to_csv("data_base.csv")
data_base = pd.read_csv("data_base.csv")
ideals = build_ideals()
data = data_base.merge(ideals, on=["iso3", "year"], how="left").drop(columns="Unnamed: 0")
data.to_csv("data.csv")
