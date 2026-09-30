#!/usr/bin/env python3
"""
Download SMEAR II Hyytiälä forest "Meteorology" data (checked/QC'd quality
only) from the SMEAR AVAA API (https://smear-backend-avaa-smear-prod.2.rahtiapp.fi/q/openapi-ui/)
and write it out as daily CSV files, then convert those into daily netCDF
files with per-variable units/long_name/description metadata:

    <outdir>/YYYY/hyytiala_YYYYMMDD.csv
    <outdir>/YYYY/hyytiala_YYYYMMDD.nc

The variable list is fetched dynamically from the API
(station="Hyytiälä", category="Meteorology") rather than hardcoded, so
newly added SMEAR variables are picked up automatically.

Data is fetched one calendar month at a time (far fewer, larger requests
than one-per-day) and then split locally into daily CSV files, which are
then converted to netCDF. Already-written days are skipped on reruns,
except for a trailing --recheck-days window, since SMEAR's "checked"
quality flag can lag behind when raw data first appears -- rerunning the
script periodically will backfill newly-checked days.

The SMEAR backend returns HTTP 500 both when a request is too large (too
many variables / too long a URL) and when a specific variable simply has
no data under the requested quality flag (e.g. as of 2026-09,
HYY_META.Cloud_layer_1..5 always 500 under quality=CHECKED even though
they work fine under quality=ANY). Both cases are handled the same way:
on failure, bisect the variable list and retry each half; a single
variable that still fails is logged and excluded for the rest of the run.

Usage
-----
    python scripts/hyytiala_smear_meteo.py
    python scripts/hyytiala_smear_meteo.py --start 2021-01-01 --end 2021-12-31
    python scripts/hyytiala_smear_meteo.py --force          # redownload + reconvert everything
    python scripts/hyytiala_smear_meteo.py --skip-netcdf    # csv only
    python scripts/hyytiala_smear_meteo.py --skip-csv       # netcdf only (from existing csv)
"""

import argparse
import calendar
import datetime
import io
import logging
import re
from pathlib import Path

import pandas as pd
import requests
import xarray as xr
from requests.adapters import HTTPAdapter, Retry

log = logging.getLogger("hyytiala_smear_meteo")

API_BASE = "https://smear-backend-avaa-smear-prod.2.rahtiapp.fi"
DEFAULT_STATION = "Hyytiälä"
DEFAULT_CATEGORY = "Meteorology"
QUALITY = "CHECKED"  # per requirements, only ever use checked/QC'd data

DEFAULT_OUTDIR = Path("/projekt1/ag_maahn/data_obs_nobackup/hyytiala/csc/meteorology")
DEFAULT_START = datetime.date(2021, 1, 1)
BATCH_SIZE = 40
DEFAULT_RECHECK_DAYS = 35


# --------------------------------------------------------------------------
# SMEAR API access
# --------------------------------------------------------------------------


def make_session():
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def get_variables(session, station=DEFAULT_STATION, category=DEFAULT_CATEGORY):
    """Fetch the list of variable-metadata dicts for a station/category.

    Raises if two variables share a short name across different tables,
    since columns are keyed by short name throughout this script.
    """
    r = session.get(
        f"{API_BASE}/search/variable",
        params={"station": station, "category": category},
        timeout=60,
    )
    r.raise_for_status()
    variables = r.json()
    if not variables:
        raise RuntimeError(
            f"no variables found for station={station!r} category={category!r}"
        )
    names = [v["name"] for v in variables]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise RuntimeError(
            f"variable name(s) {dupes} appear in more than one table; "
            "short-name column keying is ambiguous"
        )
    return variables


def get_station_info(session, station=DEFAULT_STATION):
    """Fetch station name/lat/lon/elevation for netCDF global attributes."""
    r = session.get(f"{API_BASE}/search/station", timeout=30)
    r.raise_for_status()
    candidates = [s for s in r.json() if s["name"] == station]
    if not candidates:
        raise RuntimeError(f"station {station!r} not found in /search/station")
    info = candidates[0]
    m = dict(re.findall(r"(\w+)=([^;]+);?", info["dcmiPoint"]))
    return {
        "name": info["name"],
        "longitude": float(m["east"]),
        "latitude": float(m["north"]),
        "elevation": float(m["elevation"]),
    }


def parse_timeseries_csv(text):
    """Parse a /search/timeseries/csv response into a DataFrame indexed by
    a "time" DatetimeIndex, with columns renamed from "Table.Variable" to
    just "Variable"."""
    if not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    time = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute", "Second"]])
    df = df.drop(columns=["Year", "Month", "Day", "Hour", "Minute", "Second"])
    df.columns = [c.split(".", 1)[1] for c in df.columns]
    df.index = time
    df.index.name = "time"
    return df


def fetch_timeseries_csv(session, tablevariables, frm, to, excluded, quality=QUALITY):
    """Fetch CSV data for `tablevariables` (list of "Table.Variable"
    strings) over [frm, to) (ISO-8601 strings), bisecting on any HTTP
    error to isolate/skip individual bad variables without failing the
    whole batch (see module docstring)."""
    tvs = [tv for tv in tablevariables if tv not in excluded]
    if not tvs:
        return pd.DataFrame()
    params = [("tablevariable", tv) for tv in tvs]
    params += [("from", frm), ("to", to), ("quality", quality)]
    r = session.get(f"{API_BASE}/search/timeseries/csv", params=params, timeout=180)
    if r.status_code == 200:
        return parse_timeseries_csv(r.text)
    if len(tvs) == 1:
        log.warning(
            "%s returned HTTP %s under quality=%s from %s to %s; excluding "
            "it for the rest of this run",
            tvs[0],
            r.status_code,
            quality,
            frm,
            to,
        )
        excluded.add(tvs[0])
        return pd.DataFrame()
    mid = len(tvs) // 2
    left = fetch_timeseries_csv(session, tvs[:mid], frm, to, excluded, quality)
    right = fetch_timeseries_csv(session, tvs[mid:], frm, to, excluded, quality)
    if left.empty:
        return right
    if right.empty:
        return left
    return pd.concat([left, right], axis=1)


# --------------------------------------------------------------------------
# CSV download
# --------------------------------------------------------------------------


def month_range(start, end):
    cur = datetime.date(start.year, start.month, 1)
    last = datetime.date(end.year, end.month, 1)
    while cur <= last:
        yield cur.year, cur.month
        cur = (pd.Timestamp(cur) + pd.DateOffset(months=1)).date()


def day_csv_path(outdir, day):
    return outdir / f"{day.year:04d}" / f"hyytiala_{day:%Y%m%d}.csv"


def day_nc_path(outdir, day):
    return outdir / f"{day.year:04d}" / f"hyytiala_{day:%Y%m%d}.nc"


def fetch_month(session, tablevariables, year, month, excluded):
    frm = pd.Timestamp(year=year, month=month, day=1)
    to = frm + pd.DateOffset(months=1)
    frames = []
    for i in range(0, len(tablevariables), BATCH_SIZE):
        batch = tablevariables[i : i + BATCH_SIZE]
        df = fetch_timeseries_csv(session, batch, frm.isoformat(), to.isoformat(), excluded)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    month_df = pd.concat(frames, axis=1)
    month_df = month_df.loc[:, ~month_df.columns.duplicated()]
    return month_df.sort_index()


def write_day_csv(day_df, all_var_names, path, day):
    path.parent.mkdir(parents=True, exist_ok=True)
    if day_df.empty:
        day_df = pd.DataFrame(columns=all_var_names)
        day_df.index.name = "time"
    else:
        day_df = day_df.reindex(columns=all_var_names).sort_index()
    day_df.to_csv(path, date_format="%Y-%m-%dT%H:%M:%S", float_format="%.10g")


def download_csvs(session, variables, outdir, start, end, force, recheck_days):
    all_var_names = sorted(v["name"] for v in variables)
    tablevariables = sorted(f"{v['tableName']}.{v['name']}" for v in variables)

    excluded = set()
    recheck_cutoff = datetime.date.today() - datetime.timedelta(days=recheck_days)

    for year, month in month_range(start, end):
        days_in_month = calendar.monthrange(year, month)[1]
        month_days = [
            datetime.date(year, month, d)
            for d in range(1, days_in_month + 1)
            if start <= datetime.date(year, month, d) <= end
        ]
        if not month_days:
            continue
        pending_days = [
            d
            for d in month_days
            if force or d >= recheck_cutoff or not day_csv_path(outdir, d).exists()
        ]
        if not pending_days:
            continue

        log.info("fetching %04d-%02d (%d day(s) pending)", year, month, len(pending_days))
        month_df = fetch_month(session, tablevariables, year, month, excluded)

        for d in pending_days:
            if month_df.empty:
                day_df = month_df
            else:
                day_df = month_df[
                    (month_df.index >= pd.Timestamp(d))
                    & (month_df.index < pd.Timestamp(d) + pd.Timedelta(days=1))
                ]
            write_day_csv(day_df, all_var_names, day_csv_path(outdir, d), d)
        log.info("  wrote %d file(s) to %s", len(pending_days), outdir / f"{year:04d}")

    if excluded:
        log.warning(
            "%d variable(s) never returned checked-quality data and were skipped "
            "for this whole run: %s",
            len(excluded),
            sorted(v.split(".", 1)[1] for v in excluded),
        )


# --------------------------------------------------------------------------
# netCDF conversion
# --------------------------------------------------------------------------


def build_variable_metadata(variables):
    meta = {}
    for v in variables:
        meta[v["name"]] = {
            "units": v.get("unit") or "1",
            "long_name": v.get("title") or v["name"],
            # "title" mirrors the older Hyytiälä netCDF archive's convention,
            # which (confusingly) used "title" for the long description text
            # and "long_name" for the short name; kept for compatibility with
            # code/notebooks written against those files.
            "title": v.get("description") or "",
            "description": v.get("description") or "",
            "source": v.get("source") or "",
            "smear_table": v["tableName"],
            "smear_variable": v["name"],
            "smear_period_start": v.get("periodStart") or "",
            "smear_period_end": v.get("periodEnd") or "",
        }
    return meta


def csv_to_dataset(csv_path, day, var_meta, station_info, category):
    df = pd.read_csv(csv_path, index_col="time", parse_dates=["time"])
    ds = xr.Dataset.from_dataframe(df)

    for name in ds.data_vars:
        attrs = var_meta.get(name)
        if attrs:
            ds[name].attrs.update({k: v for k, v in attrs.items() if v})
        else:
            ds[name].attrs["note"] = "no metadata returned by the SMEAR API at conversion time"

    ds["time"].attrs.update({"long_name": "time", "standard_name": "time"})

    ds.attrs.update(
        {
            "title": f"SMEAR II {station_info['name']} forest {category}, {day:%Y-%m-%d}",
            "station": station_info["name"],
            "station_longitude": station_info["longitude"],
            "station_longitude_units": "degrees_east",
            "station_latitude": station_info["latitude"],
            "station_latitude_units": "degrees_north",
            "station_elevation": station_info["elevation"],
            "station_elevation_units": "m",
            "quality": QUALITY,
            "source": "SMEAR (Station for Measuring Ecosystem-Atmosphere Relations) AVAA API",
            "source_url": API_BASE,
            "Conventions": "CF-1.8",
            "history": (
                f"created {datetime.datetime.now(datetime.timezone.utc).isoformat()} "
                f"by scripts/hyytiala_smear_meteo.py from {csv_path.name}"
            ),
        }
    )
    return ds


def day_range(start, end):
    d = start
    while d <= end:
        yield d
        d += datetime.timedelta(days=1)


def convert_to_netcdf(variables, station_info, outdir, category, start, end, overwrite):
    var_meta = build_variable_metadata(variables)

    n_converted = n_skipped_existing = n_missing_csv = 0
    for day in day_range(start, end):
        csv_path = day_csv_path(outdir, day)
        nc_path = day_nc_path(outdir, day)

        if not csv_path.exists():
            n_missing_csv += 1
            continue
        if nc_path.exists() and not overwrite:
            n_skipped_existing += 1
            continue

        ds = csv_to_dataset(csv_path, day, var_meta, station_info, category)
        nc_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
        ds.to_netcdf(nc_path, encoding=encoding)
        ds.close()
        n_converted += 1

    log.info(
        "converted %d file(s); %d already existed (skipped); %d had no CSV yet",
        n_converted,
        n_skipped_existing,
        n_missing_csv,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_date(s):
    return datetime.date.fromisoformat(s)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START)
    parser.add_argument("--end", type=parse_date, default=datetime.date.today())
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--station", default=DEFAULT_STATION)
    parser.add_argument("--category", default=DEFAULT_CATEGORY)
    parser.add_argument(
        "--force", action="store_true", help="redownload and reconvert even if output files already exist"
    )
    parser.add_argument(
        "--recheck-days",
        type=int,
        default=DEFAULT_RECHECK_DAYS,
        help="always redownload the trailing N days, since SMEAR's checked-quality "
        "flag can lag behind when raw data first appears",
    )
    parser.add_argument("--skip-csv", action="store_true", help="skip the CSV download step")
    parser.add_argument("--skip-netcdf", action="store_true", help="skip the netCDF conversion step")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.end < args.start:
        parser.error("--end must not be before --start")

    session = make_session()
    variables = get_variables(session, args.station, args.category)
    log.info(
        "found %d '%s' variable(s) for station '%s'",
        len(variables),
        args.category,
        args.station,
    )

    if not args.skip_csv:
        download_csvs(session, variables, args.outdir, args.start, args.end, args.force, args.recheck_days)

    if not args.skip_netcdf:
        station_info = get_station_info(session, args.station)
        convert_to_netcdf(
            variables, station_info, args.outdir, args.category, args.start, args.end, args.force
        )


if __name__ == "__main__":
    main()
