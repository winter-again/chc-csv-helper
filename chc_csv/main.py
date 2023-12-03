from pathlib import Path

import numpy as np
import pandas as pd
import typer
from typing_extensions import Annotated

app = typer.Typer()


@app.command()
def head(
    filename: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    n: Annotated[
        int,
        typer.Option(
            "--nrows",
            "-n",
            help="Number of rows to preview from the top via pandas.DataFrame.head()",
        ),
    ] = 5,
):
    """
    Show head of the given dataframe
    """
    df = pd.read_csv(filename)
    print(df.head(n=n))


@app.command()
def impute(
    file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
            help="CSV file to operate on.",
        ),
    ],
    cols: Annotated[str, typer.Option("--cols", "-c", help="Columns to impute.")],
    val: Annotated[
        str,
        typer.Option(
            "--val", "-v", help="The column value that should be replaced (e.g., '<=5')"
        ),
    ],
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="A random seed for reproducibility.")
    ],
    all_cause_cols: Annotated[
        str,
        typer.Option(
            "--all-cause-cols",
            "-a",
            help="All-cause columns that should be imputed first. Note that each column here should match 1:1 with a column passed to --cols (order matters).",
        ),
    ]
    | None = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=True,
            readable=False,
            resolve_path=True,
            help="Location to save the imputed result.",
        ),
    ]
    | None = None,
):
    """
    Impute columns of a CHC data export (CSV file).
    """
    df = pd.read_csv(file, dtype=str)  # read all as str
    rng = np.random.default_rng(seed)
    imp_val = int(val.replace("<=", "").strip())
    col_names = cols.split(",")

    non_exist_cols = [col for col in col_names if col not in df.columns]
    if len(non_exist_cols) != 0:
        print(
            f"Some columns specified by --cols are missing from the file:\n{', '.join(c for c in non_exist_cols)}"
        )
        raise typer.Exit(code=1)

    if all_cause_cols is not None:
        ac_col_names = all_cause_cols.split(",")
        non_exist_ac_cols = [col for col in ac_col_names if col not in df.columns]
        if len(non_exist_ac_cols) != 0:
            print(
                f"Some columns specified by --all-cause-cols are missing from the file:\n{', '.join(c for c in non_exist_ac_cols)}"
            )
            raise typer.Exit(code=1)

        if len(ac_col_names) != len(col_names):
            print(
                "Make sure the values for --cols and --all-cause-cols are of the same length and are properly ordered."
            )
            raise typer.Exit(code=1)

        for col, ac_col in zip(col_names, ac_col_names):
            df.loc[df[ac_col] == val, ac_col] = rng.integers(
                1, imp_val + 1, size=len(df.loc[df[ac_col] == val])
            )
            df.loc[df[col] == val, col] = df.apply(
                lambda x: impute_wrt_ac(x[ac_col], rng), axis=1
            )
    else:
        for col in col_names:
            df.loc[df[col] == val, col] = rng.integers(
                0, imp_val + 1, size=len(df.loc[df[col] == val])
            )

    if output is not None:
        df.to_csv(output, index=False)

    print(df)


def impute_wrt_ac(all_cause, rng):
    imp_val = rng.integers(1, float(all_cause) + 1, size=1)[0]
    return imp_val
