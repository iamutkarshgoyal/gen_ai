import pandas as pd
import re

def extract_numbers(text):
    if pd.isna(text):
        return set()
    
    text = str(text)
    numbers = set()
    
    ranges = re.findall(r'(\d+)-(\d+)', text)
    for start, end in ranges:
        numbers.update(range(int(start), int(end)+1))
    
    singles = re.findall(r'\b\d+\b', text)
    numbers.update(map(int, singles))
    return numbers

def check_missing(text):
    nums = extract_numbers(text)
    required = set(range(1,7))
    return required - nums


df = pd.read_excel("TIME TABLE TR WISE 2026-27 - Copy.xlsx", sheet_name="Table 1", skiprows=1)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

for col in df.columns:
    if col not in ["TEACHER’S NAME", "S.N"]:
        df[f"{col}_missing"] = df[col].apply(check_missing)


df.to_excel("output.xlsx", index=False)