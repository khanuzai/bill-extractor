"""Render a `UtilityBill` to PDF through one of the HTML bill templates."""

from datetime import date
from decimal import Decimal
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from weasyprint import HTML

from src.generator.data import BillExtras
from src.schema import UtilityBill

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Template names, in the order the CLI rotates through them.
TEMPLATES: tuple[str, ...] = (
    "bordered_table",
    "two_column",
    "summary_box",
    "two_page",
    "receipt",
)


def date_us(value: date) -> str:
    """03/15/2026"""
    return f"{value:%m/%d/%Y}"


def date_long(value: date) -> str:
    """March 15, 2026"""
    return f"{value:%B} {value.day}, {value.year}"


def date_abbr(value: date) -> str:
    """15-Mar-2026"""
    return f"{value:%d-%b-%Y}"


def date_iso(value: date) -> str:
    """2026-03-15"""
    return value.isoformat()


def money(value: Decimal) -> str:
    """1,234.50, with a leading minus for credits: -12.00"""
    return f"{value:,.2f}"


def money_parens(value: Decimal) -> str:
    """1,234.50, with accounting parentheses for credits: (12.00)"""
    return f"({-value:,.2f})" if value < 0 else f"{value:,.2f}"


def money_cr(value: Decimal) -> str:
    """1,234.50, with a CR suffix for credits: 12.00 CR"""
    return f"{-value:,.2f} CR" if value < 0 else f"{value:,.2f}"


def usage(value: Decimal) -> str:
    """Thousands-separated usage, keeping the decimals the meter reported: 1,047 or 34.7"""
    return f"{value:,}"


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html"]),
        undefined=StrictUndefined,
    )
    env.filters.update(
        date_us=date_us,
        date_long=date_long,
        date_abbr=date_abbr,
        date_iso=date_iso,
        money=money,
        money_parens=money_parens,
        money_cr=money_cr,
        usage=usage,
    )
    return env


_ENV = _environment()


def render_html(bill: UtilityBill, extras: BillExtras, template: str) -> str:
    """Return `bill` rendered as HTML through the named template."""
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; expected one of {TEMPLATES}")
    return _ENV.get_template(f"{template}.html").render(bill=bill, extras=extras)


def render_pdf(bill: UtilityBill, extras: BillExtras, template: str, output_path: Path) -> None:
    """Render `bill` through the named template and write the PDF to `output_path`."""
    HTML(string=render_html(bill, extras, template)).write_pdf(output_path)
