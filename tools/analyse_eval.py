"""Aggregate and print evaluation results from nav_evaluator CSV output.

Usage:
  python3 tools/analyse_eval.py /tmp/eval_results.csv
  python3 tools/analyse_eval.py results_house2.csv results_house3.csv
"""

import csv
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'method':          r['method'],
                'trial':           int(r['trial']),
                'goal_label':      r['goal_label'],
                'success':         int(r['success']),
                'collision':       int(r['collision']),
                'costmap_failure': int(r['costmap_failure']),
                'time_s':          float(r['time_s']),
                'path_dist_m':     float(r['path_dist_m']),
            })
    return rows


def summarise(rows):
    by_method = defaultdict(list)
    for r in rows:
        by_method[r['method']].append(r)

    stats = {}
    for method, rs in by_method.items():
        n = len(rs)
        stats[method] = {
            'n':               n,
            'success_pct':     100 * sum(r['success']          for r in rs) / n,
            'collision_pct':   100 * sum(r['collision']         for r in rs) / n,
            'costmap_fail_pct':100 * sum(r['costmap_failure']   for r in rs) / n,
            'avg_time_s':      sum(r['time_s']                  for r in rs) / n,
            'avg_dist_m':      sum(r['path_dist_m']             for r in rs) / n,
        }
    return stats


def print_table(stats, title=''):
    methods = list(stats.keys())
    if title:
        print(f'\n{"="*64}')
        print(f'  {title}')
        print(f'{"="*64}')

    col_w = 18
    hdr = f"{'Metric':<25}" + ''.join(f"{m:>{col_w}}" for m in methods)
    print(hdr)
    print('-' * len(hdr))

    rows = [
        ('Success rate (%)',      'success_pct',     '{:.1f}'),
        ('Collision rate (%)',    'collision_pct',    '{:.1f}'),
        ('Costmap failure (%)',   'costmap_fail_pct', '{:.1f}'),
        ('Avg time (s)',          'avg_time_s',       '{:.2f}'),
        ('Avg path dist (m)',     'avg_dist_m',       '{:.2f}'),
        ('Goals evaluated (N)',   'n',                '{:d}'),
    ]
    for label, key, fmt in rows:
        vals = ''.join(
            f"{fmt.format(stats[m][key]):>{col_w}}"
            for m in methods
        )
        print(f'{label:<25}{vals}')
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    all_rows = []
    for path in sys.argv[1:]:
        all_rows.extend(load(path))

    if not all_rows:
        print('No rows found.')
        sys.exit(1)

    stats = summarise(all_rows)
    print_table(stats, title='Navigation Evaluation — Combined Results')


if __name__ == '__main__':
    main()
