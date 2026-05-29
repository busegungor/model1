"""
gurobi_env.py — Gurobi WLS academic environment setup.

The SAA LP solver (saa_solver.py) can use Gurobi if available, but the
default fast path is the closed-form quantile solution. This module
attempts to construct a WLS academic license environment from the
.env file; if any credential is missing or gurobipy is not installed,
it returns None and the caller falls back to the quantile solver.

────────────────────────────────────────────────────────────────────
Colab origin: c_alıs_ır_son_hali_s_u_anlık.py, lines 134-148 (approx)
────────────────────────────────────────────────────────────────────
"""

import os
import warnings
from dotenv import load_dotenv

# Load .env from the project root (one level above src/)
_env_loaded = load_dotenv()


def make_gurobi_env():
    """
    Construct a Gurobi WLS academic environment.

    Returns
    -------
    gp.Env or None
        Gurobi environment if all credentials are present and gurobipy
        is installed; otherwise None (caller should fall back to the
        quantile-based SAA solver).

    Required environment variables (loaded from .env):
        WLSACCESSID
        WLSSECRET
        LICENSEID

    Note
    ----
    Returning None is the expected behaviour when no license is
    configured — saa_solver.solve_saa() handles this gracefully.
    """
    # Try to import gurobipy; if unavailable, fall back silently
    try:
        import gurobipy as gp
    except ImportError:
        warnings.warn(
            "gurobipy not installed → SAA will use the quantile solver "
            "(closed-form, mathematically identical to the LP).",
            stacklevel=2,
        )
        return None

    # Pull credentials from environment
    access_id   = os.environ.get("WLSACCESSID")
    secret      = os.environ.get("WLSSECRET")
    license_id  = os.environ.get("LICENSEID")

    if not all([access_id, secret, license_id]):
        warnings.warn(
            "Gurobi WLS credentials not found in environment "
            "(GRB_WLSACCESSID, GRB_WLSSECRET, GRB_LICENSEID). "
            "Falling back to quantile solver.",
            stacklevel=2,
        )
        return None

    # Build the WLS environment
    try:
        params = {
            "WLSACCESSID": access_id,
            "WLSSECRET":   secret,
            "LICENSEID":   int(license_id),
            "OutputFlag":  0,        # suppress license banner
        }
        env = gp.Env(params=params)
        return env
    except Exception as e:
        warnings.warn(
            f"Gurobi environment setup failed: {e}. "
            f"Falling back to quantile solver.",
            stacklevel=2,
        )
        return None