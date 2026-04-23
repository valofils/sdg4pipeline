from __future__ import annotations
from pathlib import Path
from typing import Optional
import typer
from rich import print as rprint
from rich.table import Table
from gem_pipeline.config_loader import load_control_file, filter_configs
from gem_pipeline.pipeline import run_pipeline

app = typer.Typer(name="gem-pipeline", help="GEM SDG4 Indicator Pipeline", add_completion=False)


@app.command("run")
def cmd_run(
    control: Path = typer.Option("config/control_file.csv", "--control", "-c"),
    output: Path = typer.Option("data/outputs", "--output", "-o"),
    countries: Optional[list[str]] = typer.Option(None, "--countries"),
    indicators: Optional[list[str]] = typer.Option(None, "--indicators"),
    history: Optional[Path] = typer.Option(None, "--history"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Run the full indicator pipeline."""
    rprint("[bold cyan]GEM SDG4 Pipeline[/bold cyan] | control=" + str(control))
    results = run_pipeline(
        str(control), str(output),
        list(countries) if countries else None,
        list(indicators) if indicators else None,
        dry_run,
        str(history) if history else None,
    )
    if results:
        rprint("[bold green]Done.[/bold green]")
        for sid, p in results.items():
            rprint("  " + sid + " -> " + p)
    else:
        rprint("[bold red]No surveys processed.[/bold red]")


@app.command("validate")
def cmd_validate(
    control: Path = typer.Option("config/control_file.csv", "--control", "-c"),
):
    """Validate a control file."""
    try:
        configs = load_control_file(control)
        rprint("[bold green]Valid.[/bold green] " + str(len(configs)) + " configs loaded.")
    except Exception as e:
        rprint("[bold red]Failed:[/bold red] " + str(e))
        raise typer.Exit(1)


@app.command("inventory")
def cmd_inventory(
    control: Path = typer.Option("config/control_file.csv", "--control", "-c"),
    country: Optional[str] = typer.Option(None, "--country"),
):
    """List surveys in the control file."""
    configs = load_control_file(control)
    if country:
        configs = filter_configs(configs, countries=[country])
    t = Table(title="Survey Inventory", show_lines=True)
    for col in ["ISO3", "Country", "Year", "Survey", "Type", "Format", "Indicators"]:
        t.add_column(col)
    for c in configs:
        t.add_row(
            c.country_iso3, c.country_name, str(c.survey_year),
            c.survey_name, c.survey_type, c.file_format,
            ", ".join(c.indicators),
        )
    rprint(t)




@app.command("notes")
def cmd_notes(
    control: Path = typer.Option("config/control_file.csv", "--control", "-c"),
    output: Path = typer.Option("data/outputs", "--output", "-o"),
    countries: Optional[list[str]] = typer.Option(None, "--countries"),
):
    """Generate methodological notes for processed surveys."""
    configs = load_control_file(control)
    if countries:
        configs = filter_configs(configs, countries=list(countries))
    if not configs:
        rprint("[red]No matching configs.[/red]")
        raise typer.Exit(1)
    for cfg in configs:
        note_path = Path(str(output)) / cfg.survey_id / "methodological_note.md"
        if note_path.exists():
            rprint("[green]Exists:[/green] " + str(note_path))
        else:
            rprint("[yellow]Not found:[/yellow] " + str(note_path) +
                   " — run gem-pipeline run first")


if __name__ == "__main__":
    app()
