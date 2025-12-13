# USER INPUT: Folders to look for partial results
# We now only combine results from two folders: `quantitative` and `qualitative`.
# `quantitative` holds numeric criteria/value-functions; `qualitative` holds
# qualitative assessments. There are no 'common' folders in this simplified flow.
common_folders = []
specific_folders = ["quantitative"]
qualitative_data = ["qualitative"]

# FIRST PART: Combine results from common_folders
import os
import re
import json
import pandas as pd
from collections import defaultdict

# Directory of this script; aggregated outputs will live under
# <script_dir>/aggregatedresults
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGG_DIR = os.path.join(SCRIPT_DIR, 'aggregatedresults')

# Each folder contains a criteria.csv and possibly numbered value_functions_N.csv files
# Load and concat all common folders once (assume at least one common folder exists)
combined_common_criteria = []
# collect common value_functions by index (int -> list of DataFrames)
combined_common_value_functions_by_idx = defaultdict(list)
vf_re = re.compile(r'^value_functions(?:_(\d+))?\.csv$', flags=re.IGNORECASE)
for folder in common_folders:
    criteria_path = os.path.join(SCRIPT_DIR, folder, 'criteria.csv')
    if os.path.exists(criteria_path):
        combined_common_criteria.append(pd.read_csv(criteria_path))
    # scan for numbered value_functions files in the folder
    folderp = os.path.join(SCRIPT_DIR, folder)
    if os.path.isdir(folderp):
        for fn in os.listdir(folderp):
            m = vf_re.match(fn)
            if not m:
                continue
            idx = m.group(1)
            # ignore the unnumbered base file (idx is None) as requested
            if not idx:
                continue
            vf_path = os.path.join(folderp, fn)
            try:
                df = pd.read_csv(vf_path)
                combined_common_value_functions_by_idx[int(idx)].append(df)
            except Exception:
                continue

# Concatenate collected common criteria DataFrames
if combined_common_criteria:
    combined_common_criteria_df = pd.concat(combined_common_criteria, ignore_index=True)
else:
    combined_common_criteria_df = pd.DataFrame()
# Normalize and deduplicate common criteria by 'name' when possible
try:
    if 'name' in combined_common_criteria_df.columns:
        combined_common_criteria_df['name'] = combined_common_criteria_df['name'].astype(str).str.strip()
        combined_common_criteria_df = combined_common_criteria_df.drop_duplicates(subset=['name'], keep='first', ignore_index=True)
    else:
        combined_common_criteria_df = combined_common_criteria_df.drop_duplicates(keep='first', ignore_index=True)
except Exception:
    pass
# Also collect alternatives from common folders (if present)
combined_common_alternatives = []
for folder in common_folders:
    alternatives_path = os.path.join(SCRIPT_DIR, folder, 'alternatives.csv')
    if os.path.exists(alternatives_path):
        try:
            combined_common_alternatives.append(pd.read_csv(alternatives_path))
        except Exception:
            # fallback: read as plain CSV with pandas default; if fails, skip
            pass
if combined_common_alternatives:
    combined_common_alternatives_df = pd.concat(combined_common_alternatives, ignore_index=True)
else:
    combined_common_alternatives_df = pd.DataFrame()

# Parse qualitative folder(s) early so we can apply their -E# series both in GLOBAL and per-country outputs
qual_bucket = {}
qual_alt_names = set()
try:
    from collections import defaultdict as _dd
    qual_bucket = _dd(lambda: _dd(lambda: _dd(dict)))
    for qfolder in qualitative_data:
        qpath = os.path.join(SCRIPT_DIR, qfolder, 'alternatives.csv')
        if not os.path.exists(qpath):
            continue
        try:
            qdf = pd.read_csv(qpath, dtype=str)
        except Exception:
            continue
        qdf.columns = [str(c).strip() for c in qdf.columns]
        first_col = qdf.columns[0]
        # collect alternative names present in qualitative file headers
        for alt_col in qdf.columns[1:]:
            an = str(alt_col).strip()
            if an:
                qual_alt_names.add(an)
        for _, r in qdf.iterrows():
            label = str(r[first_col]).strip()
            if not label:
                continue
            if label.lower().endswith('starting point') or 'starting point' in label.lower():
                continue
            parts = [p.strip() for p in label.split(' - ')]
            if not parts:
                continue
            last = parts[-1]
            m = re.match(r'^E\s*(\d+)$', last, flags=re.IGNORECASE)
            if not m:
                continue
            eidx = int(m.group(1))
            row_country = 'GLOBAL'
            if len(parts) >= 2:
                maybe = parts[-2]
                if re.match(r'^[A-Za-z]{2,3}$', maybe):
                    row_country = maybe.upper()
            base_indicator = parts[0]
            for alt_col in qdf.columns[1:]:
                alt_name = str(alt_col).strip()
                if not alt_name:
                    continue
                val = r.get(alt_col)
                if pd.isna(val) or str(val).strip() == '':
                    continue
                qual_bucket[row_country][base_indicator].setdefault(alt_name, {})[eidx] = val
except Exception:
    qual_bucket = {}

# (qualitative criteria and value_functions scanning is handled later, after specific_folders processing)

# SECOND PART: Read results from all specific_folders and aggregate per country
# We expect criteria 'group' values to contain a '-COUNTRY' suffix (e.g. GORPUCODE-COUNTRY)
# Value functions also may carry the same `group` convention.

# Collect country-specific rows across all specific_folders
country_criteria = {}
# collect numbered value_functions per index and per country: idx -> country_code -> list(rows)
country_value_functions_by_idx = defaultdict(lambda: defaultdict(list))
country_value_functions_unumbered = defaultdict(list)
if specific_folders:
    for folder in specific_folders:
        criteria_path = os.path.join(SCRIPT_DIR, folder, 'criteria.csv')
        if os.path.exists(criteria_path):
            df_criteria = pd.read_csv(criteria_path)
            for _, row in df_criteria.iterrows():
                group = str(row.get('group', ''))
                if '-' in group:
                    country_code = group.split('-')[-1].strip()
                    country_criteria.setdefault(country_code, []).append(row)
                else:
                    country_criteria.setdefault('GLOBAL', []).append(row)
        # scan for numbered value_functions files in this specific folder
        folderp = os.path.join(SCRIPT_DIR, folder)
        if os.path.isdir(folderp):
            for fn in os.listdir(folderp):
                m = vf_re.match(fn)
                if not m:
                    continue
                idx = m.group(1)
                vf_path = os.path.join(folderp, fn)
                try:
                    df_vf = pd.read_csv(vf_path)
                except Exception:
                    continue
                # iterate rows and distribute by group (country or GLOBAL)
                for _, row in df_vf.iterrows():
                    group = str(row.get('group', '')) if 'group' in df_vf.columns else ''
                    # By default, try to assign by explicit group suffix (GROUP-COUNTRY).
                    assigned = False
                    if group and '-' in group:
                        country_code = group.split('-')[-1].strip()
                        if idx:
                            country_value_functions_by_idx[int(idx)][country_code].append(row)
                        else:
                            country_value_functions_unumbered[country_code].append(row)
                        assigned = True
                    # If not assigned and there's a 'name' that includes a trailing ' - XX' country token,
                    # use that as the country and sanitize the name (remove the suffix) before storing.
                    if not assigned:
                        raw_name = str(row.get('name', '')).strip()
                        mname = re.match(r'^(?P<base>.+?)\s*-\s*(?P<ctry>[A-Za-z]{2,3})$', raw_name)
                        if mname:
                            base = mname.group('base').strip()
                            ctry = mname.group('ctry').upper()
                            row_copy = row.copy()
                            row_copy['name'] = base
                            if idx:
                                country_value_functions_by_idx[int(idx)][ctry].append(row_copy)
                            else:
                                country_value_functions_unumbered[ctry].append(row_copy)
                            assigned = True
                    # Fallback: store as GLOBAL
                    if not assigned:
                        if idx:
                            country_value_functions_by_idx[int(idx)]['GLOBAL'].append(row)
                        else:
                            country_value_functions_unumbered['GLOBAL'].append(row)

                # Now include qualitative folder's criteria and numbered value_functions into the country maps
                try:
                    for qfolder in qualitative_data:
                        qcrit = os.path.join(SCRIPT_DIR, qfolder, 'criteria.csv')
                        if os.path.exists(qcrit):
                            try:
                                qdfc = pd.read_csv(qcrit)
                                for _, row in qdfc.iterrows():
                                    # sanitize qualitative criteria names: remove trailing ' - E#' and optional country token
                                    row_copy = row.copy()
                                    raw_name = str(row.get('name', '')).strip()
                                    parts = [p.strip() for p in raw_name.split(' - ')] if raw_name else []
                                    if parts:
                                        last = parts[-1]
                                        if re.match(r'^E\s*\d+$', last, flags=re.IGNORECASE):
                                            # base indicator is the first part
                                            row_copy['name'] = parts[0]
                                    group = str(row_copy.get('group', ''))
                                    if '-' in group:
                                        country_code = group.split('-')[-1].strip()
                                        country_criteria.setdefault(country_code, []).append(row_copy)
                                    else:
                                        country_criteria.setdefault('GLOBAL', []).append(row_copy)
                            except Exception:
                                pass

                        folderp = os.path.join(SCRIPT_DIR, qfolder)
                        if os.path.isdir(folderp):
                            for fn in os.listdir(folderp):
                                m = vf_re.match(fn)
                                if not m:
                                    continue
                                idx = m.group(1)
                                # ignore unnumbered base files
                                if not idx:
                                    continue
                                vf_path = os.path.join(folderp, fn)
                                try:
                                    df_vf = pd.read_csv(vf_path)
                                except Exception:
                                    continue
                                for _, row in df_vf.iterrows():
                                    group = str(row.get('group', '')) if 'group' in df_vf.columns else ''
                                    assigned = False
                                    if group and '-' in group:
                                        country_code = group.split('-')[-1].strip()
                                        country_value_functions_by_idx[int(idx)][country_code].append(row)
                                        assigned = True
                                    if not assigned:
                                        raw_name = str(row.get('name', '')).strip()
                                        mname = re.match(r'^(?P<base>.+?)\s*-\s*(?P<ctry>[A-Za-z]{2,3})$', raw_name)
                                        if mname:
                                            base = mname.group('base').strip()
                                            ctry = mname.group('ctry').upper()
                                            row_copy = row.copy()
                                            row_copy['name'] = base
                                            country_value_functions_by_idx[int(idx)][ctry].append(row_copy)
                                            assigned = True
                                    if not assigned:
                                        country_value_functions_by_idx[int(idx)]['GLOBAL'].append(row)
                except Exception:
                    pass

# Determine all country codes found across specific folders (exclude GLOBAL)
country_codes = set(country_criteria.keys())
# include country codes discovered in numbered value_functions rows
for idx, by_country in country_value_functions_by_idx.items():
    for c in by_country.keys():
        country_codes.add(c)
country_codes.discard('GLOBAL')

# Ensure aggregated output dir exists
os.makedirs(AGG_DIR, exist_ok=True)

# Simple merge: concatenate `criteria.csv` from the two source folders
# (`specific_folders` and `qualitative_data`) and write the combined file.
# This follows the user's request: the aggregated `criteria.csv` should be
# the straightforward merge of the criteria files found in those folders.
wrote_agg_criteria = False
try:
    folders_to_merge = list(dict.fromkeys(specific_folders + qualitative_data))
    dfs = []
    for fld in folders_to_merge:
        p = os.path.join(SCRIPT_DIR, fld, 'criteria.csv')
        if os.path.exists(p):
            try:
                d = pd.read_csv(p, dtype=str)
                # normalize column names
                d.columns = [str(c).strip() for c in d.columns]
                dfs.append(d)
            except Exception:
                pass

    if dfs:
        try:
            merged_df = pd.concat(dfs, ignore_index=True, sort=False)
            if 'name' in merged_df.columns:
                merged_df['name'] = merged_df['name'].astype(str).str.strip()
                merged_df = merged_df.drop_duplicates(subset=['name'], keep='first', ignore_index=True)
            else:
                merged_df = merged_df.drop_duplicates(keep='first', ignore_index=True)

            # Determine polarity ('type') for each criterion by inspecting value functions
            try:
                vf_points = defaultdict(list)
                src_folders = list(set(common_folders + specific_folders + qualitative_data))
                for sf in src_folders:
                    folderp = os.path.join(SCRIPT_DIR, sf)
                    if not os.path.isdir(folderp):
                        continue
                    for fn in os.listdir(folderp):
                        m = vf_re.match(fn)
                        if not m:
                            continue
                        vf_path = os.path.join(folderp, fn)
                        try:
                            df_vf = pd.read_csv(vf_path, dtype=str)
                        except Exception:
                            continue
                        for _, r in df_vf.iterrows():
                            name = str(r.get('name', '')).strip() if 'name' in df_vf.columns else ''
                            # sanitize trailing country suffix ' - XX'
                            try:
                                mname = re.match(r'^(?P<base>.+?)\s*-\s*(?P<ctry>[A-Za-z]{2,3})$', name)
                                if mname:
                                    name = mname.group('base').strip()
                            except Exception:
                                pass
                            if not name:
                                continue
                            pts = None
                            if 'elicited_points' in df_vf.columns:
                                pts = r.get('elicited_points')
                            elif 'points' in df_vf.columns:
                                pts = r.get('points')
                            meta = None
                            if 'elicitation_meta' in df_vf.columns:
                                meta = r.get('elicitation_meta')

                            if pts is None or str(pts).strip() == '':
                                if meta:
                                    try:
                                        meta_obj = json.loads(meta)
                                        shape = meta_obj.get('shape') if isinstance(meta_obj, dict) else None
                                        if shape and name:
                                            vf_points[name].append({'shape': shape})
                                    except Exception:
                                        pass
                                continue
                            parsed = None
                            try:
                                parsed = json.loads(pts)
                            except Exception:
                                try:
                                    parsed = eval(pts)
                                except Exception:
                                    parsed = None
                            if parsed and name:
                                vf_points[name].append(parsed)

                def interpret_endpoint_relation(item):
                    # returns 1 if max endpoint y > min endpoint y (positive),
                    # -1 if min > max (negative), 0 if unknown/cannot decide
                    try:
                        if isinstance(item, dict):
                            shape = item.get('shape')
                            if isinstance(shape, str):
                                s = shape.strip().lower()
                                if s.startswith('decre') or s.startswith('down'):
                                    return -1
                                if s.startswith('incre') or s.startswith('up'):
                                    return 1
                            return 0
                        # expect list of [x,y]
                        if isinstance(item, list) and item:
                            pts = []
                            for p in item:
                                if isinstance(p, (list, tuple)) and len(p) >= 2:
                                    try:
                                        x = float(p[0])
                                        y = float(p[1])
                                        pts.append((x, y))
                                    except Exception:
                                        # try parse numbers from strings
                                        try:
                                            x = float(str(p[0]).strip())
                                            y = float(str(p[1]).strip())
                                            pts.append((x, y))
                                        except Exception:
                                            continue
                            if not pts:
                                return 0
                            pts_sorted = sorted(pts, key=lambda t: t[0])
                            y_min = pts_sorted[0][1]
                            y_max = pts_sorted[-1][1]
                            if y_max > y_min:
                                return 1
                            if y_min > y_max:
                                return -1
                            return 0
                    except Exception:
                        return 0

                types = []
                for _, row in merged_df.iterrows():
                    cname = str(row.get('name', '')).strip()
                    exprs = vf_points.get(cname, [])
                    if not exprs:
                        types.append('positive')
                        continue
                    evaluable = 0
                    positive_count = 0
                    negative_count = 0
                    for item in exprs:
                        res = interpret_endpoint_relation(item)
                        if res == 1:
                            positive_count += 1
                            evaluable += 1
                        elif res == -1:
                            negative_count += 1
                            evaluable += 1
                    if evaluable > 0 and negative_count == evaluable:
                        types.append('negative')
                    else:
                        types.append('positive')
                merged_df['type'] = types
            except Exception:
                merged_df['type'] = ['positive'] * len(merged_df)

            merged_df.to_csv(os.path.join(AGG_DIR, 'criteria.csv'), index=False)
            wrote_agg_criteria = True
        except Exception:
            wrote_agg_criteria = False
    else:
        # no criteria files found in the two folders: write a default header
        try:
            header_cols = ['name', 'group', 'min', 'max', 'unit', 'type']
            with open(os.path.join(AGG_DIR, 'criteria.csv'), 'w', encoding='utf-8', newline='') as _f:
                _f.write(','.join(header_cols) + '\n')
            wrote_agg_criteria = True
        except Exception:
            wrote_agg_criteria = False
except Exception:
    wrote_agg_criteria = False

if not country_codes:
    # No specific country codes found: write common aggregated criteria and any numbered value_functions
    # ensure criteria are deduplicated and normalized
    try:
        if 'name' in combined_common_criteria_df.columns:
            combined_common_criteria_df['name'] = combined_common_criteria_df['name'].astype(str).str.strip()
            combined_common_criteria_df = combined_common_criteria_df.drop_duplicates(subset=['name'], keep='first', ignore_index=True)
        else:
            combined_common_criteria_df = combined_common_criteria_df.drop_duplicates(keep='first', ignore_index=True)
    except Exception:
        pass
    if not wrote_agg_criteria:
        try:
            if combined_common_criteria_df.empty:
                header_cols = ['name', 'group', 'min', 'max', 'unit', 'type']
                with open(os.path.join(AGG_DIR, 'criteria.csv'), 'w', encoding='utf-8', newline='') as _f:
                    _f.write(','.join(header_cols) + '\n')
            else:
                combined_common_criteria_df.to_csv(os.path.join(AGG_DIR, 'criteria.csv'), index=False)
        except Exception:
            try:
                header_cols = ['name', 'group', 'min', 'max', 'unit', 'type']
                with open(os.path.join(AGG_DIR, 'criteria.csv'), 'w', encoding='utf-8', newline='') as _f:
                    _f.write(','.join(header_cols) + '\n')
            except Exception:
                pass
    # write each numbered common value_functions_N if present
    for idx, df_list in combined_common_value_functions_by_idx.items():
        try:
            df_common = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
            out_path = os.path.join(AGG_DIR, f'value_functions_{idx}.csv')
            if not df_common.empty:
                df_common.to_csv(out_path, index=False)
        except Exception:
            pass
else:
    # For each country, create a single aggregated result under aggregatedresults/<COUNTRY>/
    for country_code in sorted(country_codes):
        out_dir = os.path.join(AGG_DIR, country_code)
        os.makedirs(out_dir, exist_ok=True)

        # Build criteria: prefer specific rows for this country, then GLOBAL, then common defaults
        criteria_parts = [pd.DataFrame(country_criteria.get(country_code, [])), pd.DataFrame(country_criteria.get('GLOBAL', [])), combined_common_criteria_df]
        country_criteria_df = pd.concat(criteria_parts, ignore_index=True)

        # Strip "-COUNTRY" suffix for this country in group column (so group values are normalized)
        suffix_pattern = re.compile(r"-" + re.escape(country_code) + r"$")
        if 'group' in country_criteria_df.columns:
            country_criteria_df['group'] = country_criteria_df['group'].astype(str).apply(lambda g: suffix_pattern.sub('', g))

        # Deduplicate criteria rows by 'name' (trim whitespace first) so country-specific rows override common defaults
        try:
            if 'name' in country_criteria_df.columns:
                country_criteria_df['name'] = country_criteria_df['name'].astype(str).str.strip()
                country_criteria_df = country_criteria_df.drop_duplicates(subset=['name'], keep='first', ignore_index=True)
            else:
                country_criteria_df = country_criteria_df.drop_duplicates(keep='first', ignore_index=True)
        except Exception:
            pass

        # Defer writing the per-country `criteria.csv` until after we've assembled
        # the final alternatives frame so we can compute indicator min/max from
        # actual alternative values (this ensures cases where all alternatives
        # have the same value — e.g. always 0 — are captured).
        # The `country_criteria_df` DataFrame is available here and will be
        # updated later once `final_df` is built.
        
        # ---------- Combine alternatives for this country into final normalized shape ----------
        # Desired final shape: header with 'name' followed by one column per indicator (rows = alternatives)
        # Get the list of indicators for this country from the combined criteria dataframe
        indicator_names = []
        if 'name' in country_criteria_df.columns:
            indicator_names = [str(x) for x in country_criteria_df['name'].astype(str).tolist() if x and str(x).strip()]

        # helper: ensure final frame exists with proper columns
        final_cols = ['name'] + indicator_names
        final = {'df': pd.DataFrame(columns=final_cols)}

        def ensure_row(alt_name):
            alt_name = str(alt_name)
            df = final['df']
            if alt_name in df['name'].values:
                return
            # append new row with empty strings for indicators
            new = {c: '' for c in df.columns}
            new['name'] = alt_name
            final['df'] = pd.concat([df, pd.DataFrame([new])], ignore_index=True, sort=False)

        def integrate_two_col(df_src):
            # Interpret as rows: [alt_name, value] or [alt_name, ind1, ind2...]
            df = df_src.copy()
            df.columns = [str(c).strip() for c in df.columns]
            cols = list(df.columns)
            if len(cols) >= 2:
                name_col = cols[0]
                for col in cols[1:]:
                    indicator = str(col).strip()
                    for _, r in df.iterrows():
                        alt = str(r[name_col]).strip()
                        val = r[col]
                        ensure_row(alt)
                        cur = final['df']
                        if indicator not in cur.columns:
                            cur[indicator] = ''
                        cur.loc[cur['name'] == alt, indicator] = val
                        final['df'] = cur

        def integrate_indicator_rows(df_src):
            # Interpret df_src where first column contains indicator names and subsequent columns are alternatives
            df = df_src.copy()
            df.columns = [str(c).strip() for c in df.columns]
            cols = list(df.columns)
            ind_col = cols[0]
            for alt_col in cols[1:]:
                alt_name = str(alt_col).strip()
                for _, r in df.iterrows():
                    indicator = str(r[ind_col]).strip()
                    val = r[alt_col]
                    ensure_row(alt_name)
                    cur = final['df']
                    if indicator not in cur.columns:
                        cur[indicator] = ''
                    cur.loc[cur['name'] == alt_name, indicator] = val
                    final['df'] = cur

        # Build an initial set of alternative names from common alternatives
        alt_name_set = set()
        if not combined_common_alternatives_df.empty:
            try:
                # if there is a 'name' or first column that lists alternative names, use it
                first_col = combined_common_alternatives_df.columns[0]
                vals = combined_common_alternatives_df[first_col].astype(str).tolist()
                for v in vals:
                    if v and str(v).strip():
                        alt_name_set.add(str(v).strip())
            except Exception:
                pass

        # incorporate common alternatives first (if present)
        if not combined_common_alternatives_df.empty:
            try:
                integrate_two_col(combined_common_alternatives_df)
            except Exception:
                pass

        # process specific folders
        for folder in specific_folders:
            alternatives_path = os.path.join(SCRIPT_DIR, folder, 'alternatives.csv')
            if not os.path.exists(alternatives_path):
                continue
            try:
                df_src = pd.read_csv(alternatives_path, dtype=str)
                # normalize header names to avoid leading/trailing-space mismatches
                df_src.columns = [str(c).strip() for c in df_src.columns]
                # Also gather alternative names from this specific file so we don't ignore them
                try:
                    if df_src.shape[1] >= 2:
                        # if first header is 'name', values in first column are alternative names
                        if str(df_src.columns[0]).strip().lower() == 'name':
                            for v in df_src.iloc[:,0].astype(str).tolist():
                                if v and str(v).strip():
                                    alt_name_set.add(str(v).strip())
                        else:
                            # otherwise headers after the first are candidate alternative names
                            for h in df_src.columns[1:]:
                                if h and str(h).strip():
                                    alt_name_set.add(str(h).strip())
                except Exception:
                    pass
                # Special-case: files where rows are countries and columns are alternatives
                # (header first cell is the indicator name, subsequent columns are alternative names)
                try:
                    first_col_values = df_src.iloc[:, 0].astype(str).str.strip().str.upper().tolist()
                    if country_code.upper() in first_col_values:
                        # extract the row for this country and assign values for this indicator
                        row = df_src[df_src.iloc[:, 0].astype(str).str.strip().str.upper() == country_code.upper()].iloc[0]
                        indicator_name = str(df_src.columns[0])
                        # record that we hit the country-row special case
                        # (removed debug log write)
                        for alt_col in df_src.columns[1:]:
                            alt_name = str(alt_col).strip()
                            val = row[alt_col]
                            # (removed debug log write)
                            ensure_row(alt_name)
                            df = final['df']
                            if indicator_name not in df.columns:
                                df[indicator_name] = ''
                            df.loc[df['name'] == alt_name, indicator_name] = val
                            final['df'] = df
                        # we've handled this file for this country
                        continue
                except Exception:
                    pass
            except Exception:
                # parsing failed for this specific file — skip it (do not copy partial files)
                continue

            # Heuristic 1: if first header is 'name' -> rows are alternative-name centric
            first_header = str(df_src.columns[0]).strip().lower()
            try:
                if first_header == 'name':
                    try:
                        integrate_two_col(df_src)
                        continue
                    except Exception:
                        pass

                # Heuristic 2: if first header is an indicator (in our list), then rows are indicators and cols alternatives
                if df_src.shape[1] >= 2 and str(df_src.columns[0]) in indicator_names:
                    try:
                        integrate_indicator_rows(df_src)
                        continue
                    except Exception:
                        pass

                # Heuristic 3: if first column values mostly match indicators, treat file as indicator-rows
                try:
                    sample_first_col = df_src.iloc[:, 0].astype(str).tolist()
                    matches = sum(1 for v in sample_first_col if str(v) in indicator_names)
                    if matches >= max(1, int(0.4 * len(sample_first_col))):
                        try:
                            integrate_indicator_rows(df_src)
                            continue
                        except Exception:
                            pass
                except Exception:
                    pass
                # Fallback: attempt two-col integration
                try:
                    integrate_two_col(df_src)
                except Exception:
                    # fallback integration failed — skip this file
                    pass
            except Exception:
                # unexpected error while processing this file — skip it without copying
                pass

        # Materialize qualitative parsed data (if any) into this country's final dataframe
        # ensure any discovered alternative names are present as rows (include qual alternatives)
        try:
            # (debug dump removed)
            for alt in sorted(set(list(alt_name_set) + list(qual_alt_names))):
                ensure_row(alt)
        except Exception:
            pass
        try:
            for scope_key in (country_code.upper(), 'GLOBAL'):
                kb = qual_bucket.get(scope_key, {})
                for base_ind, alt_map in kb.items():
                    cur = final['df']
                    if base_ind not in cur.columns:
                        cur[base_ind] = ''
                    for alt_name, emap in alt_map.items():
                        ensure_row(alt_name)
                        values = [emap[k] for k in sorted(emap.keys())]
                        cell = json.dumps({'Discrete': [values]})
                        cur.loc[cur['name'] == alt_name, base_ind] = cell
                        final['df'] = cur
        except Exception:
            # qual_bucket may be undefined or malformed; ignore and continue
            pass

        # normalize columns: ensure 'name' first, then indicators (use union of discovered indicators)
        # Final columns should be 'name' + indicator_names (keep desired indicator order)
        merged_inds = list(dict.fromkeys(indicator_names))  # preserve order, remove duplicates
        out_cols = ['name'] + merged_inds
        # ensure at least one column for each desired name exists in final['df']
        for c in out_cols:
            if c not in final['df'].columns:
                final['df'][c] = ''

        # Coalesce any duplicate columns that share the same label (pandas allows duplicate column labels)
        try:
            # DEBUG: dump assembled final['df'] columns and first rows to help trace missing numeric values
            # (debug dump removed)
            # (debug dump removed)
            # helper to pick first non-empty value in a row sequence
            def first_nonempty_row(vals):
                for v in vals:
                    if pd.isna(v):
                        continue
                    if str(v).strip() != '':
                        return v
                return ''

            # build a new DataFrame with one column per desired out_col, coalescing duplicates
            coalesced = pd.DataFrame()
            for c in out_cols:
                # select all columns in final['df'] whose name equals c
                cols_with_name = [col for col in final['df'].columns if col == c]
                if not cols_with_name:
                    coalesced[c] = [''] * len(final['df'])
                elif len(cols_with_name) == 1:
                    coalesced[c] = final['df'][cols_with_name[0]].astype(object)
                else:
                    # coalesce row-wise
                    coalesced[c] = final['df'][cols_with_name].apply(lambda r: first_nonempty_row(r.values), axis=1)

            final_df = coalesced.copy()
            # Clean up names (strip) and drop empty-name rows
            final_df.loc[:, 'name'] = final_df['name'].astype(str).str.strip()
            final_df = final_df[final_df['name'] != '']

            # group by name to merge duplicate alternative rows taking first non-empty per column
            if not final_df.empty:
                def agg_first_nonempty(col):
                    for v in col:
                        if pd.isna(v):
                            continue
                        if str(v).strip() != '':
                            return v
                    return ''
                final_df = final_df.groupby('name', sort=False).agg(lambda col: agg_first_nonempty(col.values)).reset_index()
        except Exception:
            final_df = final['df'][out_cols].copy()

        # write final alternatives.csv
        alt_out_path = os.path.join(out_dir, 'alternatives.csv')
        try:
            # If our assembled frame is empty (some heuristics failed), synthesize a minimal frame
            if final_df.empty:
                try:
                    # build rows from discovered alternative names
                    all_alts = sorted(set(list(alt_name_set) + list(qual_alt_names)))
                    # gather explicit values from specific_folders as a fallback source
                    from collections import defaultdict as __dd
                    values_map = __dd(lambda: __dd(lambda: ''))
                    # also ingest common (simple) alternatives into values_map so their values are preferred
                    try:
                        if not combined_common_alternatives_df.empty:
                            cc = combined_common_alternatives_df.copy()
                            cc.columns = [str(c).strip() for c in cc.columns]
                            if cc.shape[1] >= 2 and str(cc.columns[0]).strip().lower() == 'name':
                                for _, rr in cc.iterrows():
                                    alt = str(rr.iloc[0]).strip()
                                    for col in cc.columns[1:]:
                                        ind = str(col).strip()
                                        values_map[alt][ind] = rr[col]
                            else:
                                try:
                                    ind_col = cc.columns[0]
                                    for _, rr in cc.iterrows():
                                        ind = str(rr[ind_col]).strip()
                                        for alt_col in cc.columns[1:]:
                                            alt = str(alt_col).strip()
                                            values_map[alt][ind] = rr[alt_col]
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    for sf in specific_folders:
                        spath = os.path.join(SCRIPT_DIR, sf, 'alternatives.csv')
                        if not os.path.exists(spath):
                            continue
                        try:
                            sdf = pd.read_csv(spath, dtype=str)
                            sdf.columns = [str(c).strip() for c in sdf.columns]
                        except Exception:
                            continue
                        try:
                            # case: rows are countries (first column values include country codes)
                            first_col_values = sdf.iloc[:, 0].astype(str).str.strip().str.upper().tolist()
                            if country_code.upper() in first_col_values:
                                crow = sdf[sdf.iloc[:, 0].astype(str).str.strip().str.upper() == country_code.upper()].iloc[0]
                                ind_name = str(sdf.columns[0]).strip()
                                for alt_col in sdf.columns[1:]:
                                    alt = str(alt_col).strip()
                                    v = crow[alt_col]
                                    values_map[alt][ind_name] = v
                                continue
                        except Exception:
                            pass
                        try:
                            # case: first header is 'name' (rows per alt)
                            if str(sdf.columns[0]).strip().lower() == 'name':
                                for _, rr in sdf.iterrows():
                                    alt = str(rr.iloc[0]).strip()
                                    for col in sdf.columns[1:]:
                                        ind = str(col).strip()
                                        values_map[alt][ind] = rr[col]
                                continue
                        except Exception:
                            pass
                        try:
                            # case: rows are indicators (first column contains indicator names)
                            if sdf.shape[1] >= 2 and str(sdf.columns[0]).strip() in merged_inds:
                                for _, rr in sdf.iterrows():
                                    ind = str(rr.iloc[0]).strip()
                                    for alt_col in sdf.columns[1:]:
                                        alt = str(alt_col).strip()
                                        values_map[alt][ind] = rr[alt_col]
                                continue
                        except Exception:
                            pass
                    synth = []
                    for alt in all_alts:
                        row = {'name': alt}
                        for base_ind in merged_inds:
                            val = ''
                            # prefer any value already present in final['df'] (from earlier integrations)
                            try:
                                if base_ind in final['df'].columns:
                                    curvals = final['df'].loc[final['df']['name'] == alt, base_ind]
                                    if not curvals.empty:
                                        v = curvals.values[0]
                                        if pd.notna(v) and str(v).strip() != '':
                                            val = v
                            except Exception:
                                pass
                            # if still empty, try explicit values from specific folders first, then qualitative
                            if val == '' or (isinstance(val, str) and val.strip() == ''):
                                try:
                                    v2 = values_map.get(alt, {}).get(base_ind, '')
                                    if v2 and str(v2).strip() != '':
                                        val = v2
                                    else:
                                        kb_country = qual_bucket.get(country_code.upper(), {})
                                        emap = kb_country.get(base_ind, {}).get(alt)
                                        if emap:
                                            values = [emap[k] for k in sorted(emap.keys())]
                                            val = json.dumps({'Discrete': [values]})
                                        else:
                                            kb_global = qual_bucket.get('GLOBAL', {})
                                            emap_g = kb_global.get(base_ind, {}).get(alt)
                                            if emap_g:
                                                values = [emap_g[k] for k in sorted(emap_g.keys())]
                                                val = json.dumps({'Discrete': [values]})
                                except Exception:
                                    pass
                            row[base_ind] = val
                        synth.append(row)
                    final_df = pd.DataFrame(synth, columns=['name'] + merged_inds)
                except Exception:
                    pass
            # Before writing alternatives, update per-country criteria min/max using
            # actual assembled alternative values so case-specific ranges (including
            # constant-zero cases) are preserved.
            try:
                # ensure country_criteria_df exists
                try:
                    country_criteria_df
                except NameError:
                    country_criteria_df = pd.DataFrame(country_criteria.get(country_code, []))

                # helper to extract numeric values from a cell
                import math
                def extract_numbers_from_cell(cell):
                    nums = []
                    try:
                        if pd.isna(cell):
                            return nums
                        s = str(cell).strip()
                        if not s:
                            return nums
                        # JSON discrete cell
                        if (s.startswith('{') or s.startswith('[')):
                            try:
                                obj = json.loads(s)
                                if isinstance(obj, dict) and 'Discrete' in obj:
                                    # Discrete is expected as a list of lists
                                    di = obj.get('Discrete')
                                    if isinstance(di, list) and di:
                                        first = di[0]
                                        for v in first:
                                            try:
                                                nums.append(float(v))
                                            except Exception:
                                                # try regex fallback
                                                import re
                                                m = re.search(r"[-+]?[0-9]*\.?[0-9]+", str(v))
                                                if m:
                                                    nums.append(float(m.group(0)))
                                else:
                                    # attempt to flatten numeric values in obj
                                    import re
                                    text = json.dumps(obj)
                                    for m in re.finditer(r"[-+]?[0-9]*\.?[0-9]+", text):
                                        nums.append(float(m.group(0)))
                                return nums
                            except Exception:
                                pass
                        # otherwise try direct float conversion
                        try:
                            nums.append(float(s))
                            return nums
                        except Exception:
                            pass
                        # fallback: regex find numbers in string
                        import re
                        for m in re.finditer(r"[-+]?[0-9]*\.?[0-9]+", s):
                            try:
                                nums.append(float(m.group(0)))
                            except Exception:
                                continue
                        return nums
                    except Exception:
                        return nums

                # iterate indicators and compute min/max from final_df
                for ind in merged_inds:
                    try:
                        if ind not in final_df.columns:
                            continue
                        col = final_df[ind].astype(object).tolist()
                        vals = []
                        for cell in col:
                            try:
                                nums = extract_numbers_from_cell(cell)
                                if nums:
                                    vals.extend(nums)
                            except Exception:
                                continue
                        if vals:
                            mn = min(vals)
                            mx = max(vals)
                            # find row in country_criteria_df and set min/max
                            try:
                                if 'name' in country_criteria_df.columns:
                                    mask = country_criteria_df['name'].astype(str).str.strip() == str(ind).strip()
                                    if mask.any():
                                        country_criteria_df.loc[mask, 'min'] = mn
                                        country_criteria_df.loc[mask, 'max'] = mx
                                    else:
                                        # append new row
                                        newr = {'name': ind, 'group': '', 'min': mn, 'max': mx}
                                        country_criteria_df = pd.concat([country_criteria_df, pd.DataFrame([newr])], ignore_index=True)
                                else:
                                    # no name column: append
                                    newr = {'name': ind, 'group': '', 'min': mn, 'max': mx}
                                    country_criteria_df = pd.concat([country_criteria_df, pd.DataFrame([newr])], ignore_index=True)
                            except Exception:
                                pass
                    except Exception:
                        pass

                # normalize and write per-country criteria now
                try:
                    if 'name' in country_criteria_df.columns:
                        country_criteria_df['name'] = country_criteria_df['name'].astype(str).str.strip()
                        country_criteria_df = country_criteria_df.drop_duplicates(subset=['name'], keep='first', ignore_index=True)
                except Exception:
                    pass
                try:
                    country_criteria_df.to_csv(os.path.join(out_dir, 'criteria.csv'), index=False)
                except Exception:
                    pass
            except Exception:
                pass

            final_df.to_csv(alt_out_path, index=False)
        except Exception:
            try:
                with open(alt_out_path, 'w', encoding='utf-8', newline='') as f:
                    f.write('name\n')
            except Exception:
                pass

        # Build numbered value_functions outputs: preserve file numbering
        # determine all indices present for this country (from common and specific)
        idxs = set(combined_common_value_functions_by_idx.keys()) | set(country_value_functions_by_idx.keys())
        for idx in sorted(idxs):
            vf_parts = []
            # include common parts for this index
            common_list = combined_common_value_functions_by_idx.get(idx, [])
            if common_list:
                try:
                    vf_parts.append(pd.concat(common_list, ignore_index=True))
                except Exception:
                    for d in common_list:
                        vf_parts.append(d)
            # include country-specific rows for this index
            specific_rows = country_value_functions_by_idx.get(idx, {}).get(country_code, [])
            if specific_rows:
                vf_parts.append(pd.DataFrame(specific_rows))
            # include GLOBAL rows matching this country's criteria names
            global_rows = country_value_functions_by_idx.get(idx, {}).get('GLOBAL', [])
            if global_rows and 'name' in country_criteria_df.columns:
                try:
                    df_global = pd.DataFrame(global_rows)
                    names_set = set(country_criteria_df['name'].astype(str).tolist())
                    if 'name' in df_global.columns:
                        matched = df_global[df_global['name'].astype(str).isin(names_set)]
                        if not matched.empty:
                            vf_parts.append(matched)
                except Exception:
                    pass
            # if we have parts, concatenate and write to value_functions_{idx}.csv
            # As a safety: also scan all source folders for any rows labelled for this country/index
            # (catch cases where earlier parsing missed them). We'll append matching rows to vf_parts.
            try:
                seen_pairs = set()
                src_folders = list(set(common_folders + specific_folders + qualitative_data))
                for sf in src_folders:
                    folderp = os.path.join(SCRIPT_DIR, sf)
                    if not os.path.isdir(folderp):
                        continue
                    # consider both numbered and unnumbered files; we only want the same index
                    for fn in os.listdir(folderp):
                        m = vf_re.match(fn)
                        if not m:
                            continue
                        fidx = m.group(1)
                        if not fidx or int(fidx) != int(idx):
                            continue
                        try:
                            df_src = pd.read_csv(os.path.join(folderp, fn), dtype=str)
                        except Exception:
                            continue
                        for _, r in df_src.iterrows():
                            # determine target country
                            group = str(r.get('group', '')).strip() if 'group' in df_src.columns else ''
                            assigned = None
                            if group and '-' in group:
                                assigned = group.split('-')[-1].strip().upper()
                            else:
                                raw_name = str(r.get('name', '')).strip()
                                mname = re.match(r'^(?P<base>.+?)\s*-\s*(?P<ctry>[A-Za-z]{2,3})$', raw_name)
                                if mname:
                                    assigned = mname.group('ctry').upper()
                            if assigned == country_code.upper():
                                # sanitize name if needed (remove trailing ' - XX')
                                row_copy = r.copy()
                                raw_name = str(r.get('name', '')).strip()
                                mname = re.match(r'^(?P<base>.+?)\s*-\s*(?P<ctry>[A-Za-z]{2,3})$', raw_name)
                                if mname:
                                    row_copy['name'] = mname.group('base').strip()
                                key = (str(row_copy.get('name','')).strip(), str(row_copy.get('group','')).strip())
                                if key in seen_pairs:
                                    continue
                                seen_pairs.add(key)
                                vf_parts.append(pd.DataFrame([row_copy]))
            except Exception:
                pass

            if vf_parts:
                try:
                    vf_out = pd.concat(vf_parts, ignore_index=True)
                except Exception:
                    # try to convert list elements to DataFrame and concat
                    try:
                        vf_out = pd.concat([pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x for x in vf_parts], ignore_index=True)
                    except Exception:
                        vf_out = None
                if vf_out is not None and not vf_out.empty:
                    # strip group suffixes if present
                    if 'group' in vf_out.columns:
                        vf_out['group'] = vf_out['group'].astype(str).apply(lambda g: suffix_pattern.sub('', g))
                # normalize name and deduplicate by name to avoid repeated indicator rows
                try:
                    if 'name' in vf_out.columns:
                        vf_out['name'] = vf_out['name'].astype(str).str.strip()
                        vf_out = vf_out.drop_duplicates(subset=['name'], keep='first', ignore_index=True)
                    else:
                        vf_out = vf_out.drop_duplicates(keep='first', ignore_index=True)
                except Exception:
                    pass
                try:
                    vf_out.to_csv(os.path.join(out_dir, f'value_functions_{idx}.csv'), index=False)
                except Exception:
                    pass