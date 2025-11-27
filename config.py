# config.py
"""
Compatibility shim for configuration loading.
Previously this module defined HEPConfig and load_from_config_hep directly.
Now we centralize logic in `_configuration_handler.py` and expose the same
helper(s) here for backward compatibility while emitting a deprecation
warning on import.
"""
import sys

# Strict deprecation stub for top-level `config` shim.
# Any import of `config` should be replaced with imports from `_configuration_handler`.
# Example: from _configuration_handler import load_config, load_from_config_hep

raise ImportError(
    "The top-level 'config' shim has been removed.\n"
    "Import directly from '_configuration_handler' instead, e.g.:\n"
    "    from _configuration_handler import load_config, load_from_config_hep\n"
    "If you need a transitional compatibility shim, update your local code to explicitly alias:\n"
    "    import _configuration_handler as config\n"
    "This stub is intentional to fail-fast and force callers to update their imports."
)
