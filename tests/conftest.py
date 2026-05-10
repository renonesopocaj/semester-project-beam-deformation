def pytest_addoption(parser):
    """
    Informal specification: config-validation tests should accept a Hydra-like config selector and
    optional overrides from CLI, so users can preflight custom config combinations before real runs.
    """
    parser.addoption("--cn", action="store", help="Hydra config name to validate for the selected test file.")
    parser.addoption(
        "--override",
        action="append",
        default=[],
        help="Hydra override for config validation; repeat the flag for multiple overrides.",
    )
