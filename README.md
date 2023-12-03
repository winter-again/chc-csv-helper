# chc-csv-helper

A simple CLI for imputing counts and all-cause columns in CHC exports (CSV files).

# Installation

Using pipx is convenient: `pipx install git+https://github.com/winter-again/chc-csv-helper`

# Commands

There are currently only two commands. See `chc-csv --help` for more details:

- `head`: A simple wrapper around `pd.DataFrame.head()` for checking a file
- `impute`: Use this to impute values in a target CSV file
