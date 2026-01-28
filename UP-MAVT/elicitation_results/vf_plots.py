"""Plot elicited value functions (overlay experts).

Folder layout expected:
  UP-MAVT/elicitation_results/<expert_id>/<COUNTRY>/value_functions.csv

Creates one plot per criterion/value function and overlays all experts.
"""

from __future__ import annotations

import ast
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable

import matplotlib


# ---------------------------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------------------------

# Select the country to evaluate.
COUNTRY = "PO"  # e.g. "IT", "CH", "FR", "PO"

# Save figures (recommended). If SHOW_PLOTS is False, we force a non-GUI backend.
SHOW_PLOTS = False

# Output settings
SAVE_FORMATS = ("pdf",)
DPI = 200
FIGSIZE = (6.5, 6.5)

# Blue-only palette (repo already uses #2b78c8 in other plots)
BLUE_PALETTE = (
	"#08306B",
	"#08519C",
	"#2171B5",
	"#2b78c8",
	"#4292C6",
	"#6BAED6",
	"#9ECAE1",
)

# Ranking heatmap sizing (standard layout)
RANK_FIG_W_PER_EXPERT = 0.8
RANK_FIG_H_PER_ALT = 0.35
RANK_FIG_MIN_W = 6.5
RANK_FIG_MIN_H = 4.0

# If True, includes confidence (when available) in legend labels.
INCLUDE_CONFIDENCE_IN_LABEL = True

# Also create ranking plots for qualitative indicators (Score-unit criteria)
# showing how alternative rankings change across experts.
PLOT_QUAL_RANKINGS = True

# If an expert did not elicit a value function for a criterion, you *can* infer
# a linear one from UP-MAVT/criteria.csv (min/max + type). Default is False to
# avoid showing curves for experts who provided no elicitation for that criterion.
INFER_LINEAR_IF_MISSING = False


# Use a non-interactive backend when not showing plots.
if not SHOW_PLOTS:
	matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CriteriaMeta:
	name: str
	group: str | None
	min_value: float | None
	max_value: float | None
	unit: str | None


@dataclass(frozen=True)
class GlobalCriteriaMeta:
	name: str
	group: str | None
	min_value: float
	max_value: float
	unit: str | None
	crit_type: str  # "positive" | "negative"


def _is_expert_dir_name(name: str) -> bool:
	return bool(re.fullmatch(r"\d+", name.strip()))


def _safe_filename(name: str) -> str:
	s = str(name).strip()
	s = re.sub(r"\s+", "_", s)
	s = re.sub(r"[^A-Za-z0-9_.-]", "_", s)
	s = re.sub(r"_+", "_", s)
	return s.strip("_") or "value_function"


def _parse_points(points_raw: str) -> list[tuple[float, float]]:
	if points_raw is None:
		raise ValueError("Missing elicited_points")

	s = str(points_raw).strip()
	if not s:
		raise ValueError("Empty elicited_points")

	try:
		parsed = json.loads(s)
	except Exception:
		parsed = ast.literal_eval(s)

	points: list[tuple[float, float]] = []
	for pair in parsed:
		if not isinstance(pair, (list, tuple)) or len(pair) != 2:
			raise ValueError(f"Invalid point: {pair!r}")
		x, y = float(pair[0]), float(pair[1])
		points.append((x, y))

	# Sort by x for plotting consistency
	points.sort(key=lambda t: t[0])
	return points


def _interp_piecewise_linear(points: list[tuple[float, float]], x: float) -> float:
	"""Piecewise-linear interpolation with clamping to endpoints."""

	if not points:
		raise ValueError("Cannot interpolate empty point set")

	xs = [p[0] for p in points]
	ys = [p[1] for p in points]
	if x <= xs[0]:
		return float(ys[0])
	if x >= xs[-1]:
		return float(ys[-1])

	for i in range(len(points) - 1):
		x0, y0 = points[i]
		x1, y1 = points[i + 1]
		if x0 <= x <= x1:
			if x1 == x0:
				return float(y1)
			t = (x - x0) / (x1 - x0)
			return float(y0 + t * (y1 - y0))

	return float(ys[-1])


def _parse_distribution_to_scalar(value: str) -> float:
	"""Parse an alternatives.csv cell to a representative scalar.

	Supports Discrete/Normal/Uniform/Trapezoidal and some repo-specific types.
	"""

	s = str(value).strip()
	if s == "" or s.lower() == "nan":
		raise ValueError("Empty alternative cell")

	# Fast path for plain numbers.
	try:
		return float(s)
	except Exception:
		pass

	# Parse dict-like strings.
	parsed = None
	try:
		parsed = json.loads(s)
	except Exception:
		parsed = ast.literal_eval(s)

	if not isinstance(parsed, dict) or not parsed:
		raise ValueError(f"Unsupported alternative cell format: {s!r}")

	# Take first (and usually only) distribution.
	dist_name = next(iter(parsed.keys()))
	params = parsed[dist_name]

	if dist_name == "Discrete":
		# Examples: [[35.15]] or [["1.0"]]
		v = params[0][0]
		return float(v)
	if dist_name == "Normal":
		return float(params[0])  # mean
	if dist_name == "Uniform":
		return 0.5 * (float(params[0]) + float(params[1]))
	if dist_name == "Trapezoidal":
		a, b, c, d = [float(x) for x in params]
		return (a + 2 * b + 2 * c + d) / 6.0
	if dist_name.startswith("Special"):
		# Repo-specific, often encoded with first element as a "central" value.
		# Example: {'Special_1': [[1], 0.3, 0.7]} -> use 1
		try:
			return float(params[0][0])
		except Exception:
			pass

	raise ValueError(f"Unsupported distribution type: {dist_name!r}")


def _load_alternatives_scores(alternatives_csv_path: str) -> dict[str, dict[str, float]]:
	"""Load alternative scores.

	Returns: criterion_name -> {alternative_name -> scalar score}

	Supports two layouts:
	  - Rows are alternatives (header includes 'name')
	  - Rows are indicators (header includes 'indicator')
	"""

	with open(alternatives_csv_path, newline="") as f:
		reader = csv.reader(f)
		rows = list(reader)
	if not rows:
		raise ValueError(f"Empty alternatives file: {alternatives_csv_path}")

	header = [h.strip() for h in rows[0]]
	if not header:
		raise ValueError(f"Invalid header in {alternatives_csv_path}")

	first = header[0].lower()
	out: dict[str, dict[str, float]] = {}

	if first == "name":
		# Alternative-oriented
		crit_names = header[1:]
		for r in rows[1:]:
			if not r:
				continue
			alt_name = str(r[0]).strip()
			if not alt_name:
				continue
			for idx, crit in enumerate(crit_names, start=1):
				if idx >= len(r):
					continue
				cell = r[idx]
				if str(cell).strip() == "":
					continue
				try:
					v = _parse_distribution_to_scalar(cell)
				except Exception:
					continue
				out.setdefault(crit, {})[alt_name] = float(v)
		return out

	if first in {"indicator", "criterion", "criteria"}:
		# Indicator-oriented
		alt_names = header[1:]
		for r in rows[1:]:
			if not r:
				continue
			crit_name = str(r[0]).strip()
			if not crit_name:
				continue
			for idx, alt in enumerate(alt_names, start=1):
				if idx >= len(r):
					continue
				cell = r[idx]
				if str(cell).strip() == "":
					continue
				try:
					v = float(cell)
				except Exception:
					try:
						v = _parse_distribution_to_scalar(cell)
					except Exception:
						continue
				out.setdefault(crit_name, {})[alt] = float(v)
		return out

	raise ValueError(
		f"Unsupported alternatives.csv layout in {alternatives_csv_path}. "
		f"Expected first column 'name' or 'indicator', got {header[0]!r}"
	)


def _load_criteria_meta(criteria_csv_path: str) -> dict[str, CriteriaMeta]:
	meta: dict[str, CriteriaMeta] = {}
	with open(criteria_csv_path, newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			name = (row.get("name") or "").strip()
			if not name:
				continue
			group = (row.get("group") or "").strip() or None
			unit = (row.get("unit") or "").strip() or None

			def _to_float(val: str | None) -> float | None:
				if val is None:
					return None
				s = str(val).strip()
				if not s:
					return None
				return float(s)

			meta[name] = CriteriaMeta(
				name=name,
				group=group,
				min_value=_to_float(row.get("min")),
				max_value=_to_float(row.get("max")),
				unit=unit,
			)
	return meta


def _load_global_criteria_meta(criteria_csv_path: str) -> dict[str, GlobalCriteriaMeta]:
	meta: dict[str, GlobalCriteriaMeta] = {}
	with open(criteria_csv_path, newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			name = (row.get("name") or "").strip()
			if not name:
				continue
			group = (row.get("group") or "").strip() or None
			unit = (row.get("unit") or "").strip() or None

			min_value = float(str(row.get("min") or "").strip())
			max_value = float(str(row.get("max") or "").strip())
			crit_type = (row.get("type") or "").strip().lower()
			if crit_type not in {"positive", "negative"}:
				raise ValueError(f"Invalid criteria type for {name!r}: {crit_type!r}")

			meta[name] = GlobalCriteriaMeta(
				name=name,
				group=group,
				min_value=min_value,
				max_value=max_value,
				unit=unit,
				crit_type=crit_type,
			)
	return meta


def _iter_expert_country_vf_paths(base_dir: str, country: str) -> Iterable[tuple[str, str]]:
	"""Yield (expert_id, vf_path) pairs.

	Some experts may have value functions stored directly under the expert folder
	(shared/qualitative indicators) in addition to (or instead of) the country
	subfolder. We include both when present.
	"""

	for entry in sorted(os.listdir(base_dir), key=lambda s: (not _is_expert_dir_name(s), s)):
		if not _is_expert_dir_name(entry):
			continue

		expert_dir = os.path.join(base_dir, entry)

		# 1) Shared VF (expert-level)
		vf_shared = os.path.join(expert_dir, "value_functions.csv")
		if os.path.isfile(vf_shared):
			yield entry, vf_shared

		# 2) Country-specific VF
		vf_country = os.path.join(expert_dir, country, "value_functions.csv")
		if os.path.isfile(vf_country):
			yield entry, vf_country


def _load_value_functions(vf_csv_path: str) -> dict[str, dict]:
	"""Return map: criterion_name -> {'points': [(x,y)], 'group': str|None, 'confidence': float|None}."""

	result: dict[str, dict] = {}
	with open(vf_csv_path, newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			name = (row.get("name") or "").strip()
			if not name:
				continue
			points = _parse_points(row.get("elicited_points") or "")
			group = (row.get("group") or "").strip() or None

			conf_val = row.get("confidence")
			confidence = None
			if conf_val is not None and str(conf_val).strip() != "":
				confidence = float(conf_val)

			result[name] = {
				"points": points,
				"group": group,
				"confidence": confidence,
			}

	if not result:
		raise ValueError(f"No value functions found in {vf_csv_path}")
	return result


_COUNTRY_SUFFIX_RE = re.compile(r"^(?P<base>.+?)\s*-\s*(?P<country>[A-Za-z]{2})\s*$")


def _normalize_criterion_name_for_country(name: str, country: str) -> str | None:
	"""Normalize country-suffixed criteria.

	Some elicitation files store criteria as e.g. "Licensing Status - IT".
	When plotting for COUNTRY=IT we want to plot that under "Licensing Status".
	Criteria suffixed for other countries are ignored.
	"""

	raw = str(name).strip()
	m = _COUNTRY_SUFFIX_RE.match(raw)
	if not m:
		return raw

	suffix_country = m.group("country").upper()
	if suffix_country != country.upper():
		return None

	return m.group("base").strip()


def _choose_xlim(meta: CriteriaMeta | None, points_per_expert: list[list[tuple[float, float]]]) -> tuple[float, float]:
	xs = [x for pts in points_per_expert for (x, _y) in pts]
	if not xs:
		return (0.0, 1.0)
	min_x, max_x = float(min(xs)), float(max(xs))

	if meta and meta.min_value is not None and meta.max_value is not None:
		# Use metadata bounds only if they match the elicited x-range.
		eps = 1e-9
		if min_x >= meta.min_value - eps and max_x <= meta.max_value + eps:
			return (float(meta.min_value), float(meta.max_value))

	# Default: use data bounds (with a tiny padding)
	if min_x == max_x:
		return (min_x - 1.0, max_x + 1.0)

	pad = 0.02 * (max_x - min_x)
	return (min_x - pad, max_x + pad)


def _is_score_like_domain(points: list[tuple[float, float]]) -> bool:
	if not points:
		return False
	xs = [x for x, _y in points]
	min_x, max_x = float(min(xs)), float(max(xs))
	# Heuristic: score scale in raw form is typically 1..6 (sometimes with fractions).
	return (min_x >= 0.9) and (max_x <= 6.1)


def _is_qualitative_score(meta: CriteriaMeta | None, gmeta: GlobalCriteriaMeta | None) -> bool:
	name = None
	if meta and meta.name:
		name = meta.name
	elif gmeta and gmeta.name:
		name = gmeta.name
	if name and name.strip().lower() == "percived safety":
		return False

	unit = None
	if meta and meta.unit:
		unit = meta.unit
	elif gmeta and gmeta.unit:
		unit = gmeta.unit
	return unit is not None and unit.strip().lower() == "score"


def _rescale_score_domain_to_unit_interval(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
	# Map [1, 6] -> [0, 1] via (x-1)/5.
	scaled: list[tuple[float, float]] = []
	for x, y in points:
		scaled.append(((float(x) - 1.0) / 5.0, float(y)))
	scaled.sort(key=lambda t: t[0])
	return scaled


def _extend_points_for_plateaus(points: list[tuple[float, float]], x_min: float, x_max: float) -> list[tuple[float, float]]:
	"""Extend points with horizontal segments at 0 or 1 beyond the elicited range.
	
	If a value function reaches 0 or 1 before x_min or after x_max, extend it
	horizontally to the domain boundary.
	"""
	if not points:
		return points
	
	extended = list(points)
	xs = [p[0] for p in points]
	ys = [p[1] for p in points]
	min_x, max_x = xs[0], xs[-1]
	
	# Check for plateau at the left boundary
	if min_x > x_min and (ys[0] == 0.0 or ys[0] == 1.0):
		extended.insert(0, (x_min, ys[0]))
	
	# Check for plateau at the right boundary
	if max_x < x_max and (ys[-1] == 0.0 or ys[-1] == 1.0):
		extended.append((x_max, ys[-1]))
	
	return sorted(extended, key=lambda t: t[0])


def plot_value_functions(country: str) -> str:
	script_dir = os.path.dirname(os.path.abspath(__file__))
	base_dir = script_dir  # UP-MAVT/elicitation_results

	# Global criteria definitions (min/max + type) for linear fallbacks.
	up_mavt_dir = os.path.dirname(base_dir)
	global_criteria_path = os.path.join(up_mavt_dir, "criteria.csv")
	global_criteria = _load_global_criteria_meta(global_criteria_path)

	expert_vf_paths = list(_iter_expert_country_vf_paths(base_dir, country))
	if not expert_vf_paths:
		raise FileNotFoundError(
			f"No value_functions.csv found for COUNTRY={country!r}. "
			f"Expected one of: {base_dir}/<expert_id>/value_functions.csv or {base_dir}/<expert_id>/{country}/value_functions.csv"
		)

	# Load criteria metadata from the first expert that has it.
	criteria_meta: dict[str, CriteriaMeta] = {}
	for expert_id, _vf_path in expert_vf_paths:
		candidate_country = os.path.join(base_dir, expert_id, country, "criteria.csv")
		candidate_shared = os.path.join(base_dir, expert_id, "criteria.csv")
		if os.path.isfile(candidate_country):
			criteria_meta = _load_criteria_meta(candidate_country)
			break
		if os.path.isfile(candidate_shared):
			criteria_meta = _load_criteria_meta(candidate_shared)
			break

	# Load and merge all experts' value functions.
	# If both shared and country-specific exist, country-specific overrides on name collisions.
	by_expert: dict[str, dict[str, dict]] = {}
	for expert_id, vf_path in expert_vf_paths:
		vf_map_raw = _load_value_functions(vf_path)
		vf_map: dict[str, dict] = {}
		for crit_name, payload in vf_map_raw.items():
			normalized = _normalize_criterion_name_for_country(crit_name, country)
			if normalized is None:
				continue
			# Normalize score-like x domains to 0..1 so they can overlay with normalized VFs.
			points = payload.get("points")
			if isinstance(points, list) and _is_score_like_domain(points):
				payload = dict(payload)
				payload["points"] = _rescale_score_domain_to_unit_interval(points)
				payload["x_domain_normalized"] = True
			vf_map[normalized] = payload
		if expert_id not in by_expert:
			by_expert[expert_id] = {}
		# Merge (later loads override previous ones).
		by_expert[expert_id].update(vf_map)

	# If requested, ensure every expert has at least a linear VF for each global criterion.
	if INFER_LINEAR_IF_MISSING:
		for expert_id in by_expert.keys():
			for crit_name, gmeta in global_criteria.items():
				if crit_name in by_expert[expert_id]:
					continue
				if gmeta.crit_type == "positive":
					pts = [(gmeta.min_value, 0.0), (gmeta.max_value, 1.0)]
				else:
					pts = [(gmeta.min_value, 1.0), (gmeta.max_value, 0.0)]
				by_expert[expert_id][crit_name] = {
					"points": pts,
					"group": gmeta.group,
					"confidence": None,
					"inferred_linear": True,
				}

	# Union of criteria names across experts (some experts may have fewer indicators)
	all_criteria = sorted({c for m in by_expert.values() for c in m.keys()})
	if not all_criteria:
		raise ValueError("No criteria found across experts")

	out_dir = os.path.join(base_dir, "plots", country)
	os.makedirs(out_dir, exist_ok=True)

	print(f"Experts found for {country}: {', '.join(sorted(by_expert.keys(), key=lambda s: int(s) if s.isdigit() else s))}")
	print(f"Criteria to plot: {len(all_criteria)}")

	cmap = plt.get_cmap("tab10")
	expert_ids_sorted = sorted(by_expert.keys(), key=lambda s: int(s) if s.isdigit() else s)
	colors = {eid: BLUE_PALETTE[i % len(BLUE_PALETTE)] for i, eid in enumerate(expert_ids_sorted)}

	for crit_name in all_criteria:
		gmeta = global_criteria.get(crit_name)
		meta = criteria_meta.get(crit_name)
		if _is_qualitative_score(meta, gmeta):
			# Skip qualitative indicators from VF overlay plots.
			continue

		fig, ax = plt.subplots(figsize=FIGSIZE)
		if hasattr(ax, "set_box_aspect"):
			ax.set_box_aspect(1)

		points_for_xlim: list[list[tuple[float, float]]] = []
		group = None

		any_score_domain_normalized = False
		for eid in expert_ids_sorted:
			crit = by_expert[eid].get(crit_name)
			if not crit:
				continue

			pts = crit["points"]
			points_for_xlim.append(pts)
			if group is None:
				group = crit.get("group")

			if crit.get("x_domain_normalized"):
				any_score_domain_normalized = True

		if not points_for_xlim:
			plt.close(fig)
			continue

		meta = criteria_meta.get(crit_name)
		x_min, x_max = _choose_xlim(meta, points_for_xlim)

		# Now plot with extended points for plateaus
		for eid in expert_ids_sorted:
			crit = by_expert[eid].get(crit_name)
			if not crit:
				continue

			pts = crit["points"]
			pts_extended = _extend_points_for_plateaus(pts, x_min, x_max)

			xs = [p[0] for p in pts_extended]
			ys = [p[1] for p in pts_extended]

			conf = crit.get("confidence")
			alpha = 0.85
			label = f"E{eid}"
			if conf is not None:
				alpha = min(0.95, max(0.35, 0.25 + 0.2 * float(conf)))
				if INCLUDE_CONFIDENCE_IN_LABEL:
					label = f"E{eid}" if float(conf).is_integer() else f"E{eid}"

			ax.plot(
				xs,
				ys,
				marker="o",
				markersize=3,
				linewidth=1.6,
				alpha=alpha,
				color=colors[eid],
				label=label,
			)

		ax.set_xlim(x_min, x_max)
		ax.set_ylim(-0.02, 1.02)
		ax.grid(True, alpha=0.25)

		# Labels
		if meta and meta.unit:
			xlabel = f"{crit_name} [{meta.unit}]"
		else:
			xlabel = crit_name
		if any_score_domain_normalized:
			xlabel = f"{xlabel} (normalized)"
		ax.set_xlabel(xlabel)
		ax.set_ylabel("Value")

		title_parts = [crit_name, f"{country}"]
		if group:
			title_parts.insert(1, str(group))
		ax.set_title(" — ".join(title_parts))
		ax.legend(loc="best", fontsize=8, frameon=True)

		fig.tight_layout()

		base_name = _safe_filename(crit_name)
		for fmt in SAVE_FORMATS:
			out_path = os.path.join(out_dir, f"{base_name}.{fmt}")
			fig.savefig(out_path, dpi=DPI)

		if SHOW_PLOTS:
			plt.show()
		else:
			plt.close(fig)

	return out_dir


def plot_qualitative_rankings(country: str) -> str:
	"""Create one value heatmap per qualitative indicator (Score-unit)."""

	script_dir = os.path.dirname(os.path.abspath(__file__))
	base_dir = script_dir
	up_mavt_dir = os.path.dirname(base_dir)
	global_criteria_path = os.path.join(up_mavt_dir, "criteria.csv")
	global_criteria = _load_global_criteria_meta(global_criteria_path)

	qualitative_criteria = [
		name for name, meta in global_criteria.items() if (meta.unit or "").strip().lower() == "score"
	]
	if not qualitative_criteria:
		raise ValueError("No qualitative (Score) criteria found in UP-MAVT/criteria.csv")

	# Re-load the per-expert value functions (already normalized) so we can apply them.
	expert_vf_paths = list(_iter_expert_country_vf_paths(base_dir, country))
	if not expert_vf_paths:
		raise FileNotFoundError(
			f"No value_functions.csv found for COUNTRY={country!r}. "
			f"Expected one of: {base_dir}/<expert_id>/value_functions.csv or {base_dir}/<expert_id>/{country}/value_functions.csv"
		)

	by_expert_vf: dict[str, dict[str, dict]] = {}
	for expert_id, vf_path in expert_vf_paths:
		# For rankings keep the originally elicited VFs; using alt-derived surrogates
		# would distort rank heatmaps.
		vf_map_raw = _load_value_functions(vf_path)
		vf_map: dict[str, dict] = {}
		for crit_name, payload in vf_map_raw.items():
			normalized = _normalize_criterion_name_for_country(crit_name, country)
			if normalized is None:
				continue
			points = payload.get("points")
			if isinstance(points, list) and _is_score_like_domain(points):
				payload = dict(payload)
				payload["points"] = _rescale_score_domain_to_unit_interval(points)
				payload["x_domain_normalized"] = True
			vf_map[normalized] = payload
		by_expert_vf.setdefault(expert_id, {}).update(vf_map)

	expert_ids = sorted(by_expert_vf.keys(), key=lambda s: int(s) if s.isdigit() else s)

	# Load alternative scores per expert.
	alt_scores_by_expert: dict[str, dict[str, dict[str, float]]] = {}
	for expert_id in expert_ids:
		expert_dir = os.path.join(base_dir, expert_id)
		alt_candidate_country = os.path.join(expert_dir, country, "alternatives.csv")
		alt_candidate_shared = os.path.join(expert_dir, "alternatives.csv")
		alt_path = alt_candidate_country if os.path.isfile(alt_candidate_country) else alt_candidate_shared
		if not os.path.isfile(alt_path):
			continue

		raw = _load_alternatives_scores(alt_path)
		normalized: dict[str, dict[str, float]] = {}
		for crit_name, mapping in raw.items():
			norm_name = _normalize_criterion_name_for_country(crit_name, country)
			if norm_name is None:
				continue
			normalized[norm_name] = mapping
		alt_scores_by_expert[expert_id] = normalized

	out_dir = os.path.join(base_dir, "plots", country, "rankings")
	os.makedirs(out_dir, exist_ok=True)

	cmap = plt.get_cmap("Blues")
	if hasattr(cmap, "copy"):
		cmap = cmap.copy()
	try:
		cmap.set_bad("#f4f7ff")
	except Exception:
		pass

	for crit_name in qualitative_criteria:
		# Collect experts that have both alt scores and VF for this criterion.
		available_experts = [
			eid
			for eid in expert_ids
			if (crit_name in by_expert_vf.get(eid, {})) and (crit_name in alt_scores_by_expert.get(eid, {}))
		]
		if len(available_experts) < 2:
			continue

		# Union of alternatives across those experts
		alt_names = sorted({a for eid in available_experts for a in alt_scores_by_expert[eid][crit_name].keys()})
		if not alt_names:
			continue

		# Compute value matrix: rows=alternatives, cols=experts
		value_matrix: list[list[float]] = []
		for alt in alt_names:
			value_matrix.append([float("nan")] * len(available_experts))

		for col_idx, eid in enumerate(available_experts):
			vf = by_expert_vf[eid][crit_name]
			points = vf["points"]
			x_norm = bool(vf.get("x_domain_normalized"))

			for row_idx, alt in enumerate(alt_names):
				if alt not in alt_scores_by_expert[eid][crit_name]:
					continue
				x = float(alt_scores_by_expert[eid][crit_name][alt])
				if x_norm:
					x = (x - 1.0) / 5.0
				v = _interp_piecewise_linear(points, x)
				value_matrix[row_idx][col_idx] = float(v)

		# Plot heatmap of values (0..1)
		fig_w = max(RANK_FIG_MIN_W, RANK_FIG_W_PER_EXPERT * len(available_experts))
		fig_h = max(RANK_FIG_MIN_H, RANK_FIG_H_PER_ALT * len(alt_names))
		fig, ax = plt.subplots(figsize=(fig_w, fig_h))

		data = value_matrix
		im = ax.imshow(
			data,
			aspect="auto",
			interpolation="nearest",
			cmap=cmap,
			vmin=0,
			vmax=1,
		)

		ax.set_xticks(list(range(len(available_experts))))
		ax.set_xticklabels([f"E{eid}" for eid in available_experts], rotation=45, ha="right")
		ax.set_yticks(list(range(len(alt_names))))
		ax.set_yticklabels(alt_names)

		ax.set_title(f"{crit_name} — {country}")

		cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
		cbar.set_label("Value (0–1)")
		cbar.ax.tick_params(labelsize=8)

		fig.tight_layout()
		base_name = _safe_filename(crit_name)
		for fmt in SAVE_FORMATS:
			fig.savefig(os.path.join(out_dir, f"{base_name}_values.{fmt}"), dpi=DPI)
		if SHOW_PLOTS:
			plt.show()
		else:
			plt.close(fig)

	return out_dir


def main() -> None:
	country = COUNTRY
	if len(sys.argv) >= 2 and sys.argv[1].strip():
		country = sys.argv[1].strip().upper()

	out_dir = plot_value_functions(country)
	print(f"Saved plots to: {out_dir}")

	if PLOT_QUAL_RANKINGS:
		rank_dir = plot_qualitative_rankings(country)
		print(f"Saved qualitative rankings to: {rank_dir}")


if __name__ == "__main__":
	main()

