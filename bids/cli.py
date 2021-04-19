import os
import json
from copy import deepcopy
from pathlib import Path
import click

from . import __version__
from .layout import BIDSLayoutIndexer, BIDSLayout
from .utils import validate_multiple as _validate_multiple

# alias -h to trigger help message
CONTEXT_SETTINGS = {'help_option_names': ['-h', '--help']}


class PathOrRegex(click.ParamType):
    "A helper Type to parse BIDSLayoutIndexer ignore/force entries"
    name = "path or m/regex/"

    def convert(self, value, param, ctx):
        import re
        if re.match(r"^m/.*/$", value):  # has form "m/<regex>/"
            value = re.compile(value[2:-1])
        return value


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(__version__, prog_name='pybids')
def cli():
    """Command-line interface for PyBIDS operations"""
    pass


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('root', type=click.Path(file_okay=False, exists=True))
@click.argument('db-path', type=click.Path(file_okay=False, resolve_path=True, exists=True))
@click.option('--derivatives', multiple=True, default=[False], show_default=True, flag_value=True,
              help="Specifies whether and/or which derivatives to index.")
@click.option('--reset-db', default=False, show_default=True, is_flag=True,
              help="Remove existing database index if present.")
@click.option('--validate/--no-validate', default=True, show_default=True,
              help="Check for BIDS compliance when indexing files.")
@click.option('--config', multiple=True,
              help="Optional name(s) of configuration file(s) to use.")
@click.option('--index-metadata/--no-index-metadata', default=False, show_default=True,
              help="Include metadata when indexing files.")
@click.option('--ignore', multiple=True, type=PathOrRegex(),
              help="Path (from root) or regex to exclude from indexing. "
                   "Regex entries need to fitted with leading 'm/' and trailing '/'.")
@click.option('--force-index', multiple=True, type=PathOrRegex(),
              help="Path (from root) or regex to include when indexing. "
                   "Regex entries need to fitted with leading 'm/' and trailing '/'.")
@click.option('--config-filename', type=click.Path(),
              default="layout_config.json", show_default=True,
              help="Name of filename within directories that contains configuration information.")
def layout(
    root,
    db_path,
    derivatives,
    reset_db,
    validate,
    config,
    index_metadata,
    ignore,
    force_index,
    config_filename,
):
    """
    Initialize a BIDSLayout, and create an SQLite database index.
    """

    # ensure empty multiples are set to None
    derivatives = _validate_multiple(derivatives, retval=False)
    config = _validate_multiple(config)
    ignore = _validate_multiple(ignore)
    force_index = _validate_multiple(force_index)

    if not (Path(db_path) / 'layout_index.sqlite').exists():
        reset_db = True

    layout = BIDSLayout(
        root,
        database_path=db_path,
        reset_database=reset_db,
        validate=validate,
        config=config,
        indexer=BIDSLayoutIndexer(
            validate=validate,
            index_metadata=index_metadata,
            ignore=ignore,
            force_index=force_index,
            config_filename=config_filename,
        ),
    )
    if reset_db:
        click.echo("Successfully generated database index at {}".format(db_path))
    else:
        click.echo(
            "Previously generated database index found at {}. "
            "To generate a new index, rerun with ``--reset-db``".format(db_path)
        )


@cli.command(context_settings=CONTEXT_SETTINGS)
@click.argument('root', type=click.Path(file_okay=False, exists=True))
def upgrade(root):
    """
    Upgrade common experimental BIDS features to finalized versions.
    """
    description_path = Path(root) / "dataset_description.json"
    description = json.loads(description_path.read_text())
    orig = deepcopy(description)

    # Always update DatasetType if missing
    if "DatasetType" not in description:
        val = click.prompt("Is this dataset [r]aw or [d]erivative?", default="r",
                           type=click.Choice(("r", "d"), case_sensitive=False))
        description["DatasetType"] = "raw" if val == "R" else "derivative"
    dstype = description["DatasetType"]

    if dstype == "raw":
        click.echo("No other upgrades for raw datasets at present.")
        return
    elif dstype == "derivative":
        if "PipelineDescription" in description:
            val = click.prompt("Convert PipelineDescription to GeneratedBy?", default="Y",
                               type=click.Choice("YN"))
            if val == "Y":
                description["GeneratedBy"] = [description.pop("PipelineDescription")]

    if description != orig:
        description_path.write_text(json.dumps(description))

    val = click.prompt("Load dataset and update filenames?", default="Y",
                       type=click.Choice("YN"))
    if val == "N":
        return

    layout = BIDSLayout(root, validate=False, config="bids" if dstype == "raw" else "derivatives")

    # Rename regressors.tsv to timeseries.tsv
    regressors = layout.get(suffix="regressors")
    policy = None
    with click.progressbar(regressors) as bar:
        for bidsfile in bar:
            action = policy
            new_path = bidsfile.path.replace("regressors.", "timeseries.")
            if action is None:
                action = click.prompt(
                    f"Rename {bidsfile.path} to {new_path}? ([y]es/[n]o/[A]ll/[N]one)", default="y"
                    type=click.Choice("ynAN"), show_choices=False)
                if action in "AN":
                    policy = action
            if action in "yA":
                click.echo(f"Renaming {bidsfile.path} -> {new_path}")
                os.rename(bidsfile.path, new_path)
            else:
                click.echo(f"Not renaming {bidsfile.path}")
