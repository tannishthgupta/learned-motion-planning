"""Benchmarking harness and result aggregation."""

from .analysis import (SUMMARY_FIELDS, common_solved_instances, markdown_tables,
                       print_summary, summarise, write_markdown, write_summary_csv)
from .runner import (BenchmarkConfig, read_convergence, read_rows, run_benchmark,
                     write_convergence, write_rows, write_run_metadata)

__all__ = [
    "BenchmarkConfig", "run_benchmark", "write_rows", "read_rows",
    "write_run_metadata", "write_convergence", "read_convergence",
    "summarise", "common_solved_instances", "write_summary_csv",
    "write_markdown", "markdown_tables", "print_summary", "SUMMARY_FIELDS",
]
