"""Official VINS vio.csv (ns, xyz, wxyz, velocity) -> TUM T_WI (s, xyz, xyzw)."""
import argparse
import csv
from decimal import Decimal
import math
from pathlib import Path


def convert(source, target):
    result = []
    previous = None
    with open(source, newline='') as handle:
        for row in csv.reader(handle):
            if not row or row[0].lstrip().startswith('#'):
                continue
            if len(row) < 8:
                raise ValueError('Expected VINS CSV: ns, x,y,z, qw,qx,qy,qz, ...')
            timestamp = Decimal(row[0])
            if not timestamp.is_finite() or timestamp != timestamp.to_integral_value():
                raise ValueError('VINS timestamp must be finite integer nanoseconds')
            if previous is not None and timestamp <= previous:
                raise ValueError('Non-increasing trajectory: possible estimator restart')
            values = list(map(float, row[1:8]))
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Non-finite pose')
            norm = math.sqrt(sum(v*v for v in values[3:7]))
            if norm < 1e-8:
                raise ValueError('Zero quaternion')
            xyz = values[:3]
            qw, qx, qy, qz = [v/norm for v in values[3:7]]
            seconds = timestamp / Decimal(1000000000)
            result.append(format(seconds, '.9f') + ' ' + ' '.join(
                format(v, '.12g') for v in xyz + [qx, qy, qz, qw]))
            previous = timestamp
    if len(result) < 2:
        raise ValueError('VINS did not produce at least two valid poses; inspect estimator.log')
    Path(target).write_text('# T_WI: timestamp_s tx ty tz qx qy qz qw\n' + '\n'.join(result) + '\n')
    return len(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('target')
    args = parser.parse_args()
    print('Converted', convert(args.source, args.target), 'IMU poses')
