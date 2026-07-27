"""Obtain the frozen LAQN snapshot used in the paper.

The LAQN NO2 endpoint is a LIVE public API whose station set and values drift
over time, so querying it directly does NOT reproduce the paper (a fresh query
now returns a different set of stations). By default this script therefore
downloads the exact frozen 52-node snapshot (2024-06-15 NO2) -- the same two
files that ship in ``graph_gp/data_cache/`` -- from a pinned remote mirror.
Use ``--live`` only to reconstruct a fresh, non-reproducing snapshot from the
live API.

    python fetch_laqn_cache.py          # download the frozen snapshot (default)
    python fetch_laqn_cache.py --live   # rebuild from the live API (drifts)

Produces two files in data_cache/:
    laqn_sites.json          -- site metadata (code, name, type, lat, lon)
    laqn_NO2_2024-06-15.json -- daily-mean NO2 per site (53 stations -> 52 nodes)
"""

import argparse
import json
from pathlib import Path

import requests

CACHE_DIR = Path('data_cache')
FILES = ('laqn_sites.json', 'laqn_NO2_2024-06-15.json')

# Optional mirror of the frozen snapshot. The SAME two files ship in
# graph_gp/data_cache/ (the primary, authoritative source); this remote is only
# a fallback for a checkout that is missing the cache. To enable it, set this to
# a permanent public archive of the two JSON files (e.g. a Zenodo record with a
# DOI). Left blank by default -- the shipped data_cache/ is what reproduces.
REMOTE_CACHE_BASE = ''

SITE_TYPE_ORDER = {
    'Rural': 0, 'Suburban': 1, 'Urban Background': 2,
    'Industrial': 3, 'Roadside': 4, 'Kerbside': 5,
}


def download_frozen():
    """Download the pinned frozen snapshot (the reproducible default)."""
    if not REMOTE_CACHE_BASE:
        raise SystemExit(
            'REMOTE_CACHE_BASE is not set. The frozen snapshot ships in '
            'graph_gp/data_cache/ -- restore those two files, or set '
            'REMOTE_CACHE_BASE to a public mirror of them.')
    CACHE_DIR.mkdir(exist_ok=True)
    for fn in FILES:
        url = f'{REMOTE_CACHE_BASE}/{fn}'
        print(f'Downloading frozen {fn}\n  from {url}')
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        (CACHE_DIR / fn).write_bytes(r.content)
    values = json.load(open(CACHE_DIR / 'laqn_NO2_2024-06-15.json'))
    print(f'  saved {len(values)} station values '
          f'(frozen snapshot -> 52 connected nodes)')
    if len(values) != 53:
        print(f'  [WARN] expected 53 station values, got {len(values)}; this '
              f'mirror may not be the frozen paper snapshot.')
    print('\nDone. The frozen snapshot is in data_cache/ (reproduces the paper).')


def fetch_live():
    """Reconstruct a snapshot from the live LAQN API. NON-REPRODUCING: the
    endpoint drifts, so this generally differs from the paper's 52-node data."""
    CACHE_DIR.mkdir(exist_ok=True)
    print('[--live] Reconstructing from the live LAQN API. This does NOT '
          'reproduce the paper; keep the shipped frozen cache instead.')

    # ---- Sites ----
    print('Fetching sites...')
    url = ('https://api.erg.ic.ac.uk/AirQuality/Information/'
           'MonitoringSites/GroupName=London/Json')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    raw = r.json()['Sites']['Site']

    sites = []
    for s in raw:
        if s.get('@DateClosed'):
            continue
        lat = float(s.get('@Latitude', 0))
        lon = float(s.get('@Longitude', 0))
        if lat == 0 or lon == 0:
            continue
        sites.append({
            'code': s['@SiteCode'],
            'name': s['@SiteName'],
            'type': s['@SiteType'],
            'lat': lat,
            'lon': lon,
            'type_num': SITE_TYPE_ORDER.get(s['@SiteType'], 2),
        })
    with open(CACHE_DIR / 'laqn_sites.json', 'w') as f:
        json.dump(sites, f, indent=2)
    print(f'  Saved {len(sites)} sites')

    # ---- NO2 data ----
    species, date, end_date = 'NO2', '2024-06-15', '2024-06-16'
    print(f'Fetching {species} data for {date}...')
    values = {}
    for i, site in enumerate(sites):
        code = site['code']
        try:
            url = (f'https://api.erg.ic.ac.uk/AirQuality/Data/SiteSpecies/'
                   f'SiteCode={code}/SpeciesCode={species}/'
                   f'StartDate={date}/EndDate={end_date}/Json')
            r = requests.get(url, timeout=10)
            records = r.json().get('RawAQData', {}).get('Data', [])
            vals = [float(d['@Value']) for d in records
                    if d.get('@Value') and d['@Value'] != '']
            if len(vals) >= 6:
                values[code] = sum(vals) / len(vals)
        except Exception as e:
            print(f'  [{i+1}/{len(sites)}] {code}: error ({e})')
    with open(CACHE_DIR / f'laqn_{species}_{date}.json', 'w') as f:
        json.dump(values, f, indent=2)
    print(f'  Saved {len(values)} stations with data '
          f'(live snapshot; NOT the frozen 52-node paper data)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--live', action='store_true',
                    help='reconstruct from the live LAQN API instead of '
                         'downloading the frozen snapshot; the API drifts, so '
                         'this will NOT reproduce the paper')
    args = ap.parse_args()
    if args.live:
        fetch_live()
    else:
        download_frozen()
