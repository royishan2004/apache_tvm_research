"""Predict TVM schedule knobs from shape features using saved LightGBM models."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import pandas as pd

LOGGER = logging.getLogger("predict_knobs")

RESEARCH_DIR = Path(__file__).resolve().parents[3]
REPO_DIR = Path(__file__).resolve().parents[4]
DEFAULT_MODELS_DIR = RESEARCH_DIR / "results" / "ml_schedule_predictor" / "models"
DEFAULT_BULK_OUTPUT_PATH = RESEARCH_DIR / "results" / "ml_schedule_predictor" / "predicted_knobs_all_shapes.json"
DEFAULT_UPLOAD_TIMEOUT = 120

ALLOWED_VECTOR_WIDTHS = [8, 16, 32]
ALLOWED_UNROLL_FACTORS = [64, 128, 256, 512]

DEFAULT_KNOBS = {
    "vector_width": 8,
    "unroll_factor": 128,
    "cache_write_used": 1,
    "reduction_decompose_used": 1,
}


def _load_bert_shape_helpers() -> Tuple[List[int], Dict[str, Any]]:
    """Load kernel shape helpers with a fallback for direct script execution."""
    try:
        from research.workloads.bert.bert_shapes import (  # pylint: disable=import-outside-toplevel
            M_LIST,
            mlp_compressed_shape,
            mlp_expanded_shape,
            qkv_shape,
        )
    except ModuleNotFoundError:
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))
        from research.workloads.bert.bert_shapes import (  # type: ignore  # pylint: disable=import-outside-toplevel
            M_LIST,
            mlp_compressed_shape,
            mlp_expanded_shape,
            qkv_shape,
        )

    return M_LIST, {
        "qkv": qkv_shape,
        "mlp_expand": mlp_expanded_shape,
        "mlp_reduce": mlp_compressed_shape,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _canonical_kernel_type(kernel_name: str) -> str:
    if not kernel_name:
        return "unknown"
    return str(kernel_name).strip().lower()


def build_feature_row(kernel_name: str, M: int, K: int, N: int) -> pd.DataFrame:
    """Build a one-row feature DataFrame that matches training features."""
    kernel_type = _canonical_kernel_type(kernel_name)
    denominator = (M * K) + (K * N) + (M * N)

    row = {
        "kernel_type": kernel_type,
        "M": int(M),
        "K": int(K),
        "N": int(N),
        "M_div_8": _safe_ratio(M, 8),
        "N_div_8": _safe_ratio(N, 8),
        "K_div_8": _safe_ratio(K, 8),
        "M_div_16": _safe_ratio(M, 16),
        "N_div_16": _safe_ratio(N, 16),
        "arithmetic_intensity_proxy": _safe_ratio(M * K * N, denominator),
        "reduction_ratio": _safe_ratio(K, max(N, 1)),
        "output_size": int(M * N),
        "flops": int(2 * M * K * N),
    }
    return pd.DataFrame([row])


def _nearest_allowed(value: float, candidates: Iterable[int]) -> int:
    values = list(candidates)
    return min(values, key=lambda item: abs(float(item) - float(value)))


def _clamp_vector_width(value: Any) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = DEFAULT_KNOBS["vector_width"]
    return _nearest_allowed(numeric, ALLOWED_VECTOR_WIDTHS)


def _clamp_unroll(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(DEFAULT_KNOBS["unroll_factor"])

    bounded = max(min(numeric, max(ALLOWED_UNROLL_FACTORS)), min(ALLOWED_UNROLL_FACTORS))
    return _nearest_allowed(bounded, ALLOWED_UNROLL_FACTORS)


def _as_binary_flag(value: Any, default: int) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return int(default)
    return int(numeric >= 0.5)


@lru_cache(maxsize=1)
def _load_models(models_dir: str) -> Dict[str, Dict[str, Any]]:
    base = Path(models_dir)
    artifacts = {
        "vector_width": base / "vector_width_model.pkl",
        "unroll_factor": base / "unroll_model.pkl",
        "cache_write_used": base / "cache_write_model.pkl",
        "reduction_decompose_used": base / "decompose_model.pkl",
    }

    loaded: Dict[str, Dict[str, Any]] = {}
    for key, path in artifacts.items():
        if not path.exists():
            LOGGER.warning("Model file missing for %s: %s", key, path)
            continue
        try:
            payload = joblib.load(path)
        except Exception as err:  # pylint: disable=broad-except
            LOGGER.warning("Failed to load model file for %s: %s", key, err)
            continue

        if not isinstance(payload, dict):
            LOGGER.warning("Invalid model payload for %s: %s", key, path)
            continue
        if "model" not in payload and "constant_value" not in payload:
            LOGGER.warning("Model payload missing model/constant for %s: %s", key, path)
            continue
        loaded[key] = payload

    return loaded


def _align_features(feature_row: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    encoded = pd.get_dummies(feature_row, columns=["kernel_type"], dummy_na=False)
    aligned = encoded.reindex(columns=feature_columns, fill_value=0.0)
    return aligned


def predict_schedule_knobs(kernel_name: str, M: int, K: int, N: int) -> Dict[str, int]:
    """Predict schedule knobs for a single matmul workload shape.

    Returns a safe dict and never raises to avoid breaking tuning scripts.
    """
    knobs = dict(DEFAULT_KNOBS)
    feature_row = build_feature_row(kernel_name=kernel_name, M=M, K=K, N=N)

    try:
        models = _load_models(str(DEFAULT_MODELS_DIR))
    except Exception as err:  # pylint: disable=broad-except
        LOGGER.warning("Model loading failed, using defaults: %s", err)
        return knobs

    for knob_name, payload in models.items():
        try:
            model_kind = str(payload.get("model_kind", ""))
            if model_kind.startswith("constant_"):
                prediction = payload.get("constant_value", knobs.get(knob_name))
            else:
                model = payload.get("model")
                if model is None:
                    raise ValueError("model payload does not contain a trained model")
                feature_columns = payload["feature_columns"]
                aligned = _align_features(feature_row, feature_columns)
                prediction = model.predict(aligned)[0]
            knobs[knob_name] = prediction
        except Exception as err:  # pylint: disable=broad-except
            LOGGER.warning("Prediction failed for %s, using default: %s", knob_name, err)

    knobs["vector_width"] = _clamp_vector_width(knobs.get("vector_width"))
    knobs["unroll_factor"] = _clamp_unroll(knobs.get("unroll_factor"))
    knobs["cache_write_used"] = _as_binary_flag(
        knobs.get("cache_write_used"),
        default=DEFAULT_KNOBS["cache_write_used"],
    )
    knobs["reduction_decompose_used"] = _as_binary_flag(
        knobs.get("reduction_decompose_used"),
        default=DEFAULT_KNOBS["reduction_decompose_used"],
    )

    return {
        "vector_width": int(knobs["vector_width"]),
        "unroll_factor": int(knobs["unroll_factor"]),
        "cache_write_used": int(knobs["cache_write_used"]),
        "reduction_decompose_used": int(knobs["reduction_decompose_used"]),
    }


def predict_schedule_knobs_for_all_shapes() -> pd.DataFrame:
    """Predict knobs for all known kernels and M values from bert_shapes."""
    m_list, kernels = _load_bert_shape_helpers()

    rows: List[Dict[str, int]] = []
    for kernel_name, shape_fn in kernels.items():
        for m_value in m_list:
            M, K, N = shape_fn(int(m_value))
            prediction = predict_schedule_knobs(kernel_name=kernel_name, M=M, K=K, N=N)
            rows.append(
                {
                    "kernel": kernel_name,
                    "M": int(M),
                    "K": int(K),
                    "N": int(N),
                    **prediction,
                }
            )

    return pd.DataFrame(rows)


def _format_table(dataframe: pd.DataFrame) -> str:
    """Return a presentable table string for terminal output."""
    try:
        from tabulate import tabulate  # pylint: disable=import-outside-toplevel

        return tabulate(dataframe, headers="keys", tablefmt="github", showindex=False)
    except Exception:  # pylint: disable=broad-except
        return dataframe.to_string(index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict schedule knobs for one workload or all known BERT shapes"
    )
    parser.add_argument("--kernel", help="Kernel name (qkv, mlp_expand, mlp_reduce)")
    parser.add_argument("--M", type=int, help="M dimension")
    parser.add_argument("--K", type=int, help="K dimension")
    parser.add_argument("--N", type=int, help="N dimension")
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Predict for all kernels and M values from bert_shapes.py",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON output path. If omitted with --all-shapes, defaults to "
            f"{DEFAULT_BULK_OUTPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Do not upload all-shapes predictions to data_aggregator",
    )
    parser.add_argument(
        "--upload-url",
        default=None,
        help=(
            "Override DATA_AGGREGATOR_BEST_SCHEDULE_PREDICTIONS_URL for --all-shapes upload"
        ),
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Override DATA_AGGREGATOR_PROFILE for --all-shapes upload",
    )
    parser.add_argument(
        "--upload-timeout",
        type=int,
        default=DEFAULT_UPLOAD_TIMEOUT,
        help="Upload timeout in seconds for --all-shapes upload (default: 120)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def _upload_all_shapes_predictions(
    records: List[Dict[str, int]],
    profile: Optional[str],
    upload_url: Optional[str],
    timeout: Optional[int],
) -> bool:
    """Upload all-shapes prediction rows with a local-path import fallback."""
    try:
        from research.workloads.common.data_aggregator_client import (  # pylint: disable=import-outside-toplevel
            upload_best_schedule_predictions,
        )
    except ModuleNotFoundError:
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))
        from research.workloads.common.data_aggregator_client import (  # type: ignore  # pylint: disable=import-outside-toplevel
            upload_best_schedule_predictions,
        )

    return upload_best_schedule_predictions(
        records,
        url=upload_url,
        profile=profile,
        dedupe=True,
        timeout=timeout,
    )


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.all_shapes:
        if any(value is not None for value in (args.kernel, args.M, args.K, args.N)):
            LOGGER.warning("Ignoring --kernel/--M/--K/--N because --all-shapes was specified")

        dataframe = predict_schedule_knobs_for_all_shapes()
        output_path: Optional[Path] = args.output_json or DEFAULT_BULK_OUTPUT_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)

        records = dataframe.to_dict(orient="records")
        # Overwrite on every run to keep this artifact current and deterministic.
        with output_path.open("w", encoding="utf-8") as file_out:
            json.dump(records, file_out, indent=2)

        LOGGER.info("Saved %d predictions to %s", len(dataframe), output_path)
        if args.no_upload:
            LOGGER.info("Skipping cloud upload (--no-upload set)")
        else:
            upload_ok = _upload_all_shapes_predictions(
                records=records,
                profile=args.profile,
                upload_url=args.upload_url,
                timeout=args.upload_timeout,
            )
            if upload_ok:
                LOGGER.info(
                    "Uploaded %d predictions to data_aggregator best_schedule_predictions",
                    len(records),
                )
            else:
                LOGGER.warning(
                    "Prediction upload failed; JSON file is still saved at %s",
                    output_path,
                )

        table_columns = [
            "kernel",
            "M",
            "K",
            "N",
            "vector_width",
            "unroll_factor",
            "cache_write_used",
            "reduction_decompose_used",
        ]
        print(_format_table(dataframe[table_columns]))
        return 0

    missing_args = [name for name, value in {
        "--kernel": args.kernel,
        "--M": args.M,
        "--K": args.K,
        "--N": args.N,
    }.items() if value is None]
    if missing_args:
        raise SystemExit(
            "Missing required arguments for single-shape prediction: "
            + ", ".join(missing_args)
            + ". Use --all-shapes for bulk prediction."
        )

    prediction = predict_schedule_knobs(
        kernel_name=args.kernel,
        M=args.M,
        K=args.K,
        N=args.N,
    )
    print(prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
