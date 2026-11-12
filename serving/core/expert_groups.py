"""CPU/PIM execution groups, independent of DRAM channel timing models."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PIMGroups:
    count: int = 1
    packages_per_group: int = 16
    channels_per_package: int = 2
    capacity_gib_per_group: float = 256
    bandwidth_tbps_per_package: float = 0.6
    granularity: int = 1
    attachment: str = 'direct'

    def __post_init__(self):
        if min(self.count, self.packages_per_group, self.channels_per_package, self.granularity) < 1:
            raise ValueError('PIM group dimensions and granularity must be positive')
        if self.capacity_gib_per_group <= 0 or self.bandwidth_tbps_per_package <= 0:
            raise ValueError('PIM group capacity and bandwidth must be positive')
        if self.attachment not in ('direct', 'network'):
            raise ValueError('PIM attachment must be direct or network')
        if self.attachment == 'direct' and self.count != 1:
            raise ValueError('direct attachment supports one CPU/PIM group')

    @property
    def bandwidth_bytes_per_ns(self):
        return self.packages_per_group * self.bandwidth_tbps_per_package * 1000

    @property
    def capacity_bytes(self):
        return int(self.capacity_gib_per_group * 2**30)

    def runtime_ns(self, expert_counts, weight_bytes):
        quanta = sum(math.ceil(count / self.granularity) for count in expert_counts if count)
        return math.ceil(quanta * weight_bytes / self.bandwidth_bytes_per_ns)

    def to_dict(self):
        return dict(asdict(self), bandwidth_tbps_per_group=self.bandwidth_bytes_per_ns / 1000,
                    timing='original analytical equations; CPU and internal TP overhead excluded')

    @classmethod
    def from_config(cls, config):
        return cls(**config)
