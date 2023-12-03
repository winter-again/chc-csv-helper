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


# TODO: do we need to worry about the column types and whether they're int/float or string?
# for example don't want to read FIPS col incorrectly then save it incorrectly too
# should we just read everything as string then?
# what other args of read_csv() should we allow manipulation of?
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
    all_cause_cols: Annotated[
        str,
        typer.Option(
            "--all-cause-cols",
            "-a",
            help="All-cause columns that should be imputed first. Note that each column here should match 1:1 with a column passed to --cols.",
        ),
    ]
    | None = None,
    seed: Annotated[
        int, typer.Option("--seed", "-s", help="A random seed for reproducibility.")
    ] = 123,
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
    df = pd.read_csv(file)
    rng = np.random.default_rng(seed)
    imp_val = int(val.replace("<=", "").strip())
    col_names = cols.split(",")

    if all_cause_cols is not None:
        ac_col_names = all_cause_cols.split(",")
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
        # TODO: fix this check for whether the columns passed all exist
        #
        # if ~set(col_names).issubset(df.columns):
        #     print("Some columns are missing from the file...")
        #     print(col_names)
        #     raise typer.Exit(code=1)
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
