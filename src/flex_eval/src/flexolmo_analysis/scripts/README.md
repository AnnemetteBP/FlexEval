# Scripts

These are the main entry points.

- `run_analysis.sh`
  - shell wrapper in the same style as project-level eval scripts
- `run_analysis_vllm.sh`
  - same idea, but for vLLM-oriented workflows
- `run_capture.sh`
  - shell wrapper for capture-only runs
- `run_capture_vllm.sh`
  - vLLM-flavored capture wrapper
- `run_reports.sh`
  - shell wrapper for reports-only runs
- `run_reports_vllm.sh`
  - vLLM-flavored reports wrapper
- the shell scripts call the Python package entrypoints under `src/flexolmo_analysis/cli/`
- they set `PYTHONPATH` so the bundle works as a self-contained folder
