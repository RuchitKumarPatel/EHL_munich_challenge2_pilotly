from __future__ import annotations


class MinMaxCalibrator:
    def __init__(self) -> None:
        self.minimum = 0.0
        self.maximum = 1.0

    def fit(self, values: list[float]) -> "MinMaxCalibrator":
        if values:
            self.minimum = min(values)
            self.maximum = max(values)
        return self

    def transform(self, value: float) -> float:
        if self.maximum <= self.minimum:
            return 0.5
        return max(0.0, min(1.0, (value - self.minimum) / (self.maximum - self.minimum)))

