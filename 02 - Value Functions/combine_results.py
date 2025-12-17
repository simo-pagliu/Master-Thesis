# USER INPUT: Folders to look for partial results
# We now only combine results from two folders: `quantitative` and `qualitative`.
# `quantitative` holds numeric criteria/value-functions; `qualitative` holds
# qualitative assessments. There are no 'common' folders in this simplified flow.
common_folders = []
specific_folders = ["quantitative"]
qualitative_data = ["qualitative"]
specific_codes = ['IT', 'FR', 'CH', 'PO'] 

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

# Each folder contains a criteria.csv and (now) only an unnumbered value_functions.csv
# Load and concat all common folders once (assume at least one common folder exists)
combined_common_criteria = []
for folder in common_folders:
    criteria_path = os.path.join(SCRIPT_DIR, folder, 'criteria.csv')
    if os.path.exists(criteria_path):
        combined_common_criteria.append(pd.read_csv(criteria_path))

# Concatenate collected common criteria DataFrames
if combined_common_criteria:
    combined_common_criteria_df = pd.concat(combined_common_criteria, ignore_index=True)
else:
    combined_common_criteria_df = pd.DataFrame()
# Normalize and deduplicate common criteria by 'name' when possible
if 'name' in combined_common_criteria_df.columns:
    combined_common_criteria_df['name'] = combined_common_criteria_df['name'].astype(str).str.strip()
    combined_common_criteria_df = combined_common_criteria_df.drop_duplicates(subset=['name'], keep='first', ignore_index=True)
else:
    combined_common_criteria_df = combined_common_criteria_df.drop_duplicates(keep='first', ignore_index=True)
# Also collect alternatives from common folders (if present)
combined_common_alternatives = []
for folder in common_folders:
    alternatives_path = os.path.join(SCRIPT_DIR, folder, 'alternatives.csv')
    if os.path.exists(alternatives_path):
        combined_common_alternatives.append(pd.read_csv(alternatives_path))
if combined_common_alternatives:
    combined_common_alternatives_df = pd.concat(combined_common_alternatives, ignore_index=True)
else:
    combined_common_alternatives_df = pd.DataFrame()

# Parse qualitative folder(s) early so we can apply their -E# series both in GLOBAL and per-country outputs
qual_bucket = {}
qual_alt_names = set()
from collections import defaultdict as _dd
qual_bucket = _dd(lambda: _dd(lambda: _dd(dict)))
for qfolder in qualitative_data:
    qpath = os.path.join(SCRIPT_DIR, qfolder, 'alternatives.csv')
    if not os.path.exists(qpath):
        continue
    qdf = pd.read_csv(qpath, dtype=str)
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

# (qualitative criteria and value_functions scanning is handled later, after specific_folders processing)

# SECOND PART: Read results from all specific_folders and aggregate per country
# We expect criteria 'group' values to contain a '-COUNTRY' suffix (e.g. GORPUCODE-COUNTRY)
# Value functions also may carry the same `group` convention.

# Collect country-specific rows across all specific_folders
country_criteria = {}
# collect unnumbered value_functions per country
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
        # load only unnumbered value_functions.csv from this folder
        vf_path = os.path.join(SCRIPT_DIR, folder, 'value_functions.csv')
        if os.path.exists(vf_path):
            df_vf = pd.read_csv(vf_path)
            for _, row in df_vf.iterrows():
                group = str(row.get('group', '')) if 'group' in df_vf.columns else ''
                assigned = False
                if group and '-' in group:
                    country_code = group.split('-')[-1].strip()
                    country_value_functions_unumbered[country_code].append(row)
                    assigned = True
                if not assigned:
                    raw_name = str(row.get('name', '')).strip()
                    mname = re.match(r'^(?P<base>.+?)\s*-\s*(?P<ctry>[A-Za-z]{2,3})$', raw_name)
                    if mname:
                        base = mname.group('base').strip()
                        ctry = mname.group('ctry').upper()
                        row_copy = row.copy()
                        row_copy['name'] = base
                        country_value_functions_unumbered[ctry].append(row_copy)
                        assigned = True
                if not assigned:
                    country_value_functions_unumbered['GLOBAL'].append(row)

        # Include qualitative folder's criteria and unnumbered value_functions into the country maps
        for qfolder in qualitative_data:
            qcrit = os.path.join(SCRIPT_DIR, qfolder, 'criteria.csv')
            if os.path.exists(qcrit):
                qdfc = pd.read_csv(qcrit)
                for _, row in qdfc.iterrows():
                    row_copy = row.copy()
                    raw_name = str(row.get('name', '')).strip()
                    parts = [p.strip() for p in raw_name.split(' - ')] if raw_name else []
                    if parts:
                        last = parts[-1]
                        if re.match(r'^E\s*\d+$', last, flags=re.IGNORECASE):
                            row_copy['name'] = parts[0]
                    group = str(row_copy.get('group', ''))
                    if '-' in group:
                        country_code = group.split('-')[-1].strip()
                        country_criteria.setdefault(country_code, []).append(row_copy)
                    else:
                        country_criteria.setdefault('GLOBAL', []).append(row_copy)

            qvf = os.path.join(SCRIPT_DIR, qfolder, 'value_functions.csv')
            if os.path.exists(qvf):
                qdf_vf = pd.read_csv(qvf)
                for _, row in qdf_vf.iterrows():
                    group = str(row.get('group', '')) if 'group' in qdf_vf.columns else ''
                    assigned = False
                    if group and '-' in group:
                        country_code = group.split('-')[-1].strip()
                        country_value_functions_unumbered[country_code].append(row)
                        assigned = True
                    if not assigned:
                        raw_name = str(row.get('name', '')).strip()
                        mname = re.match(r'^(?P<base>.+?)\s*-\s*(?P<ctry>[A-Za-z]{2,3})$', raw_name)
                        if mname:
                            base = mname.group('base').strip()
                            ctry = mname.group('ctry').upper()
                            row_copy = row.copy()
                            row_copy['name'] = base
                            country_value_functions_unumbered[ctry].append(row_copy)
                            assigned = True
                    if not assigned:
                        country_value_functions_unumbered['GLOBAL'].append(row)

# Use explicit country codes provided by the user at the top
country_codes = set([str(c).upper() for c in specific_codes])

# Ensure aggregated output dir exists
os.makedirs(AGG_DIR, exist_ok=True)

# Simple merge: concatenate `criteria.csv` from the two source folders
# (`specific_folders` and `qualitative_data`) and write the combined file.
# This follows the user's request: the aggregated `criteria.csv` should be
# the straightforward merge of the criteria files found in those folders.
wrote_agg_criteria = False
folders_to_merge = list(dict.fromkeys(specific_folders + qualitative_data))
dfs = []
for fld in folders_to_merge:
    p = os.path.join(SCRIPT_DIR, fld, 'criteria.csv')
    if os.path.exists(p):
        d = pd.read_csv(p, dtype=str)
        d.columns = [str(c).strip() for c in d.columns]
        dfs.append(d)

if dfs:
    merged_df = pd.concat(dfs, ignore_index=True, sort=False)
    if 'name' in merged_df.columns:
        merged_df['name'] = merged_df['name'].astype(str).str.strip()
        merged_df = merged_df.drop_duplicates(subset=['name'], keep='first', ignore_index=True)
    else:
        merged_df = merged_df.drop_duplicates(keep='first', ignore_index=True)

    # Determine polarity ('type') for each criterion by inspecting unnumbered value functions
    vf_points = defaultdict(list)
    src_folders = list(set(common_folders + specific_folders + qualitative_data))
    for sf in src_folders:
        vf_path = os.path.join(SCRIPT_DIR, sf, 'value_functions.csv')
        if not os.path.exists(vf_path):
            continue
        df_vf = pd.read_csv(vf_path, dtype=str)
        for _, r in df_vf.iterrows():
            name = str(r.get('name', '')).strip() if 'name' in df_vf.columns else ''
            mname = re.match(r'^(?P<base>.+?)\s*-\s*(?P<ctry>[A-Za-z]{2,3})$', name)
            if mname:
                name = mname.group('base').strip()
            if not name:
                continue
            pts = r.get('elicited_points') if 'elicited_points' in df_vf.columns else r.get('points')
            meta = r.get('elicitation_meta') if 'elicitation_meta' in df_vf.columns else None
            if pts is None or str(pts).strip() == '':
                if meta:
                    meta_obj = json.loads(meta)
                    shape = meta_obj.get('shape') if isinstance(meta_obj, dict) else None
                    if shape and name:
                        vf_points[name].append({'shape': shape})
                continue
            try:
                parsed = json.loads(pts)
            except Exception:
                parsed = None
            if parsed and name:
                vf_points[name].append(parsed)

    def interpret_endpoint_relation(item):
        if isinstance(item, dict):
            shape = item.get('shape')
            if isinstance(shape, str):
                s = shape.strip().lower()
                if s.startswith('decre') or s.startswith('down'):
                    return -1
                if s.startswith('incre') or s.startswith('up'):
                    return 1
            return 0
        if isinstance(item, list) and item:
            pts = []
            for p in item:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    x = float(p[0])
                    y = float(p[1])
                    pts.append((x, y))
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
        return 0

    types = []
    for _, row in merged_df.iterrows():
        cname = str(row.get('name', '')).strip()
        exprs = vf_points.get(cname, [])
        if not exprs:
            types.append('positive')
            continue
        evaluable = 0
        negative_count = 0
        for item in exprs:
            res = interpret_endpoint_relation(item)
            if res != 0:
                evaluable += 1
                if res == -1:
                    negative_count += 1
        if evaluable > 0 and negative_count == evaluable:
            types.append('negative')
        else:
            types.append('positive')
    merged_df['type'] = types

    merged_df.to_csv(os.path.join(AGG_DIR, 'criteria.csv'), index=False)
    wrote_agg_criteria = True

if not country_codes:
    # No specific country codes found: write global unnumbered value_functions if present
    vf_frames = []
    for fld in list(dict.fromkeys(specific_folders + qualitative_data)):
        p = os.path.join(SCRIPT_DIR, fld, 'value_functions.csv')
        if os.path.exists(p):
            vf_frames.append(pd.read_csv(p))
    if vf_frames:
        vf_out = pd.concat(vf_frames, ignore_index=True)
        if 'name' in vf_out.columns:
            vf_out['name'] = vf_out['name'].astype(str).str.strip()
            # sanitize trailing country suffix ' - XX'
            vf_out['name'] = vf_out['name'].apply(lambda s: re.sub(r"\s*-\s*[A-Za-z]{2,3}$", "", s))
            vf_out = vf_out.drop_duplicates(subset=['name'], keep='first', ignore_index=True)
        else:
            vf_out = vf_out.drop_duplicates(keep='first', ignore_index=True)
        vf_out.to_csv(os.path.join(AGG_DIR, 'value_functions.csv'), index=False)
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
        if 'name' in country_criteria_df.columns:
            country_criteria_df['name'] = country_criteria_df['name'].astype(str).str.strip()
            country_criteria_df = country_criteria_df.drop_duplicates(subset=['name'], keep='first', ignore_index=True)
        else:
            country_criteria_df = country_criteria_df.drop_duplicates(keep='first', ignore_index=True)

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
            # if there is a 'name' or first column that lists alternative names, use it
            first_col = combined_common_alternatives_df.columns[0]
            vals = combined_common_alternatives_df[first_col].astype(str).tolist()
            for v in vals:
                if v and str(v).strip():
                    alt_name_set.add(str(v).strip())

        # incorporate common alternatives first (if present)
        if not combined_common_alternatives_df.empty:
            integrate_two_col(combined_common_alternatives_df)

        # process specific folders
        for folder in specific_folders:
            alternatives_path = os.path.join(SCRIPT_DIR, folder, 'alternatives.csv')
            if not os.path.exists(alternatives_path):
                continue
            df_src = pd.read_csv(alternatives_path, dtype=str)
            # normalize header names to avoid leading/trailing-space mismatches
            df_src.columns = [str(c).strip() for c in df_src.columns]
            # Also gather alternative names from this specific file so we don't ignore them
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
            # Special-case: files where rows are countries and columns are alternatives
            # (header first cell is the indicator name, subsequent columns are alternative names)
            first_col_values = df_src.iloc[:, 0].astype(str).str.strip().str.upper().tolist()
            if country_code.upper() in first_col_values:
                # extract the row for this country and assign values for this indicator
                row = df_src[df_src.iloc[:, 0].astype(str).str.strip().str.upper() == country_code.upper()].iloc[0]
                indicator_name = str(df_src.columns[0])
                for alt_col in df_src.columns[1:]:
                    alt_name = str(alt_col).strip()
                    val = row[alt_col]
                    ensure_row(alt_name)
                    df = final['df']
                    if indicator_name not in df.columns:
                        df[indicator_name] = ''
                    df.loc[df['name'] == alt_name, indicator_name] = val
                    final['df'] = df
                continue

            # Heuristic 1: if first header is 'name' -> rows are alternative-name centric
            first_header = str(df_src.columns[0]).strip().lower()
            if first_header == 'name':
                integrate_two_col(df_src)
                continue

            # Heuristic 2: if first header is an indicator (in our list), then rows are indicators and cols alternatives
            if df_src.shape[1] >= 2 and str(df_src.columns[0]) in indicator_names:
                integrate_indicator_rows(df_src)
                continue

            # Heuristic 3: if first column values mostly match indicators, treat file as indicator-rows
            sample_first_col = df_src.iloc[:, 0].astype(str).tolist()
            matches = sum(1 for v in sample_first_col if str(v) in indicator_names)
            if matches >= max(1, int(0.4 * len(sample_first_col))):
                integrate_indicator_rows(df_src)
                continue
            # Fallback: attempt two-col integration
            integrate_two_col(df_src)

        # Materialize qualitative parsed data (if any) into this country's final dataframe
        # ensure any discovered alternative names are present as rows (include qual alternatives)
        for alt in sorted(set(list(alt_name_set) + list(qual_alt_names))):
            ensure_row(alt)
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

        # normalize columns: ensure 'name' first, then indicators (use union of discovered indicators)
        # Final columns should be 'name' + indicator_names (keep desired indicator order)
        merged_inds = list(dict.fromkeys(indicator_names))  # preserve order, remove duplicates
        out_cols = ['name'] + merged_inds
        # ensure at least one column for each desired name exists in final['df']
        for c in out_cols:
            if c not in final['df'].columns:
                final['df'][c] = ''

        # Coalesce any duplicate columns that share the same label (pandas allows duplicate column labels)
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
            cols_with_name = [col for col in final['df'].columns if col == c]
            if not cols_with_name:
                coalesced[c] = [''] * len(final['df'])
            elif len(cols_with_name) == 1:
                coalesced[c] = final['df'][cols_with_name[0]].astype(object)
            else:
                coalesced[c] = final['df'][cols_with_name].apply(lambda r: first_nonempty_row(r.values), axis=1)

        final_df = coalesced.copy()
        final_df.loc[:, 'name'] = final_df['name'].astype(str).str.strip()
        final_df = final_df[final_df['name'] != '']
        if not final_df.empty:
            def agg_first_nonempty(col):
                for v in col:
                    if pd.isna(v):
                        continue
                    if str(v).strip() != '':
                        return v
                return ''
            final_df = final_df.groupby('name', sort=False).agg(lambda col: agg_first_nonempty(col.values)).reset_index()

        # write final alternatives.csv
        alt_out_path = os.path.join(out_dir, 'alternatives.csv')
        # If our assembled frame is empty (some heuristics failed), synthesize a minimal frame
        if final_df.empty:
            all_alts = sorted(set(list(alt_name_set) + list(qual_alt_names)))
            from collections import defaultdict as __dd
            values_map = __dd(lambda: __dd(lambda: ''))
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
                    ind_col = cc.columns[0]
                    for _, rr in cc.iterrows():
                        ind = str(rr[ind_col]).strip()
                        for alt_col in cc.columns[1:]:
                            alt = str(alt_col).strip()
                            values_map[alt][ind] = rr[alt_col]
            for sf in specific_folders:
                spath = os.path.join(SCRIPT_DIR, sf, 'alternatives.csv')
                if not os.path.exists(spath):
                    continue
                sdf = pd.read_csv(spath, dtype=str)
                sdf.columns = [str(c).strip() for c in sdf.columns]
                first_col_values = sdf.iloc[:, 0].astype(str).str.strip().str.upper().tolist()
                if country_code.upper() in first_col_values:
                    crow = sdf[sdf.iloc[:, 0].astype(str).str.strip().str.upper() == country_code.upper()].iloc[0]
                    ind_name = str(sdf.columns[0]).strip()
                    for alt_col in sdf.columns[1:]:
                        alt = str(alt_col).strip()
                        v = crow[alt_col]
                        values_map[alt][ind_name] = v
                    continue
                if str(sdf.columns[0]).strip().lower() == 'name':
                    for _, rr in sdf.iterrows():
                        alt = str(rr.iloc[0]).strip()
                        for col in sdf.columns[1:]:
                            ind = str(col).strip()
                            values_map[alt][ind] = rr[col]
                    continue
                if sdf.shape[1] >= 2 and str(sdf.columns[0]).strip() in merged_inds:
                    for _, rr in sdf.iterrows():
                        ind = str(rr.iloc[0]).strip()
                        for alt_col in sdf.columns[1:]:
                            alt = str(alt_col).strip()
                            values_map[alt][ind] = rr[alt_col]
                    continue
            synth = []
            for alt in all_alts:
                row = {'name': alt}
                for base_ind in merged_inds:
                    val = ''
                    if base_ind in final['df'].columns:
                        curvals = final['df'].loc[final['df']['name'] == alt, base_ind]
                        if not curvals.empty:
                            v = curvals.values[0]
                            if pd.notna(v) and str(v).strip() != '':
                                val = v
                    if val == '' or (isinstance(val, str) and val.strip() == ''):
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
                    row[base_ind] = val
                synth.append(row)
            final_df = pd.DataFrame(synth, columns=['name'] + merged_inds)
            # Before writing alternatives, update per-country criteria min/max using
            # actual assembled alternative values so case-specific ranges (including
            # constant-zero cases) are preserved.
            # ensure country_criteria_df exists
            country_criteria_df
            # helper to extract numeric values from a cell
            import math
            import re as _re
            def extract_numbers_from_cell(cell):
                nums = []
                if pd.isna(cell):
                    return nums
                s = str(cell).strip()
                if not s:
                    return nums
                if (s.startswith('{') or s.startswith('[')):
                    obj = json.loads(s)
                    if isinstance(obj, dict) and 'Discrete' in obj:
                        di = obj.get('Discrete')
                        if isinstance(di, list) and di:
                            first = di[0]
                            for v in first:
                                nums.append(float(v))
                    else:
                        text = json.dumps(obj)
                        for m in _re.finditer(r"[-+]?[0-9]*\.?[0-9]+", text):
                            nums.append(float(m.group(0)))
                    return nums
                try:
                    nums.append(float(s))
                    return nums
                except Exception:
                    pass
                for m in _re.finditer(r"[-+]?[0-9]*\.?[0-9]+", s):
                    nums.append(float(m.group(0)))
                return nums

            # iterate indicators and compute min/max from final_df
            for ind in merged_inds:
                if ind not in final_df.columns:
                    continue
                col = final_df[ind].astype(object).tolist()
                vals = []
                for cell in col:
                    nums = extract_numbers_from_cell(cell)
                    if nums:
                        vals.extend(nums)
                if vals:
                    mn = min(vals)
                    mx = max(vals)
                    if 'name' in country_criteria_df.columns:
                        mask = country_criteria_df['name'].astype(str).str.strip() == str(ind).strip()
                        if mask.any():
                            country_criteria_df.loc[mask, 'min'] = mn
                            country_criteria_df.loc[mask, 'max'] = mx
                        else:
                            newr = {'name': ind, 'group': '', 'min': mn, 'max': mx}
                            country_criteria_df = pd.concat([country_criteria_df, pd.DataFrame([newr])], ignore_index=True)
                    else:
                        newr = {'name': ind, 'group': '', 'min': mn, 'max': mx}
                        country_criteria_df = pd.concat([country_criteria_df, pd.DataFrame([newr])], ignore_index=True)

            # normalize and write per-country criteria now
            if 'name' in country_criteria_df.columns:
                country_criteria_df['name'] = country_criteria_df['name'].astype(str).str.strip()
                country_criteria_df = country_criteria_df.drop_duplicates(subset=['name'], keep='first', ignore_index=True)
            country_criteria_df.to_csv(os.path.join(out_dir, 'criteria.csv'), index=False)

            final_df.to_csv(alt_out_path, index=False)

        # Write unnumbered value_functions if they exist for this country
        unnumbered_rows = country_value_functions_unumbered.get(country_code, [])
        global_unnumbered_rows = country_value_functions_unumbered.get('GLOBAL', [])
        if unnumbered_rows or global_unnumbered_rows:
            vf_unnumbered = []
            if unnumbered_rows:
                vf_unnumbered.append(pd.DataFrame(unnumbered_rows))
            if global_unnumbered_rows and 'name' in country_criteria_df.columns:
                df_global = pd.DataFrame(global_unnumbered_rows)
                names_set = set(country_criteria_df['name'].astype(str).tolist())
                if 'name' in df_global.columns:
                    matched = df_global[df_global['name'].astype(str).isin(names_set)]
                    if not matched.empty:
                        vf_unnumbered.append(matched)
            if vf_unnumbered:
                vf_out = pd.concat(vf_unnumbered, ignore_index=True)
                if 'group' in vf_out.columns:
                    vf_out['group'] = vf_out['group'].astype(str).apply(lambda g: suffix_pattern.sub('', g))
                if 'name' in vf_out.columns:
                    vf_out['name'] = vf_out['name'].astype(str).str.strip()
                    vf_out = vf_out.drop_duplicates(subset=['name'], keep='first', ignore_index=True)
                else:
                    vf_out = vf_out.drop_duplicates(keep='first', ignore_index=True)
                vf_out.to_csv(os.path.join(out_dir, 'value_functions.csv'), index=False)