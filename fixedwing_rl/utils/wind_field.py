"""Utilities for loading and querying 3D wind fields from ERA5 data."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from netCDF4 import Dataset


@dataclass
class WindField3D:
    """Represents a spatially indexed 3D wind field.

    The class loads ERA5 (or any similar) NetCDF files containing `u` and `v`
    wind components (east-west, north-south) and an optional vertical component
    `w`. The interpolation is performed in latitude, longitude, altitude, and
    optionally time, enabling realistic gusting conditions within JSBSim.
    """

    nc_path: str
    variable_u: str = "u10"
    variable_v: str = "v10"
    variable_w: Optional[str] = None
    time_variable: str = "time"
    altitude_variable: Optional[str] = None
    latitude_variable: str = "latitude"
    longitude_variable: str = "longitude"

    def __post_init__(self) -> None:
        self._ds = Dataset(self.nc_path, "r")
        self._lats = np.array(self._ds[self.latitude_variable][:])
        self._lons = np.array(self._ds[self.longitude_variable][:])
        self._alts = (
            np.array(self._ds[self.altitude_variable][:])
            if self.altitude_variable and self.altitude_variable in self._ds
            else np.array([0.0])
        )
        self._times = (
            np.array(self._ds[self.time_variable][:])
            if self.time_variable and self.time_variable in self._ds
            else np.array([0.0])
        )
        if self._times.ndim > 1:
            self._times = self._times.squeeze()

    def close(self) -> None:
        try:
            self._ds.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def _wrap_lon(self, lon: float) -> float:
        lon = lon % 360.0
        return lon if lon <= 180.0 else lon - 360.0

    def _interp_index(self, values: np.ndarray, target: float) -> Tuple[int, int, float]:
        idx = np.searchsorted(values, target) - 1
        idx = np.clip(idx, 0, len(values) - 2)
        frac = (target - values[idx]) / (values[idx + 1] - values[idx])
        frac = np.clip(frac, 0.0, 1.0)
        return idx, idx + 1, float(frac)

    def _linear_interp(self, data: np.ndarray, lat: float, lon: float, alt: float, time: float) -> float:
        lat_idx0, lat_idx1, lat_frac = self._interp_index(self._lats, lat)
        lon_idx0, lon_idx1, lon_frac = self._interp_index(self._lons, lon)
        alt_idx0, alt_idx1, alt_frac = self._interp_index(self._alts, alt)
        time_idx0, time_idx1, time_frac = self._interp_index(self._times, time)

        c0000 = data[time_idx0, alt_idx0, lat_idx0, lon_idx0]
        c0001 = data[time_idx0, alt_idx0, lat_idx0, lon_idx1]
        c0010 = data[time_idx0, alt_idx0, lat_idx1, lon_idx0]
        c0011 = data[time_idx0, alt_idx0, lat_idx1, lon_idx1]
        c0100 = data[time_idx0, alt_idx1, lat_idx0, lon_idx0]
        c0101 = data[time_idx0, alt_idx1, lat_idx0, lon_idx1]
        c0110 = data[time_idx0, alt_idx1, lat_idx1, lon_idx0]
        c0111 = data[time_idx0, alt_idx1, lat_idx1, lon_idx1]
        c1000 = data[time_idx1, alt_idx0, lat_idx0, lon_idx0]
        c1001 = data[time_idx1, alt_idx0, lat_idx0, lon_idx1]
        c1010 = data[time_idx1, alt_idx0, lat_idx1, lon_idx0]
        c1011 = data[time_idx1, alt_idx0, lat_idx1, lon_idx1]
        c1100 = data[time_idx1, alt_idx1, lat_idx0, lon_idx0]
        c1101 = data[time_idx1, alt_idx1, lat_idx0, lon_idx1]
        c1110 = data[time_idx1, alt_idx1, lat_idx1, lon_idx0]
        c1111 = data[time_idx1, alt_idx1, lat_idx1, lon_idx1]

        c00 = c0000 * (1 - lon_frac) + c0001 * lon_frac
        c01 = c0010 * (1 - lon_frac) + c0011 * lon_frac
        c10 = c0100 * (1 - lon_frac) + c0101 * lon_frac
        c11 = c0110 * (1 - lon_frac) + c0111 * lon_frac
        c0 = c00 * (1 - lat_frac) + c01 * lat_frac
        c1 = c10 * (1 - lat_frac) + c11 * lat_frac
        c_alt0 = c0 * (1 - alt_frac) + c1 * alt_frac

        c00_t = c1000 * (1 - lon_frac) + c1001 * lon_frac
        c01_t = c1010 * (1 - lon_frac) + c1011 * lon_frac
        c10_t = c1100 * (1 - lon_frac) + c1101 * lon_frac
        c11_t = c1110 * (1 - lon_frac) + c1111 * lon_frac
        c0_t = c00_t * (1 - lat_frac) + c01_t * lat_frac
        c1_t = c10_t * (1 - lat_frac) + c11_t * lat_frac
        c_alt1 = c0_t * (1 - alt_frac) + c1_t * alt_frac

        value = c_alt0 * (1 - time_frac) + c_alt1 * time_frac
        return float(value)

    def vector_at(
        self,
        latitude: float,
        longitude: float,
        altitude_m: float,
        when: Optional[dt.datetime] = None,
    ) -> Tuple[float, float, float]:
        """Return interpolated wind vector in NED coordinates (m/s)."""
        lon = self._wrap_lon(longitude)
        lat = np.clip(latitude, self._lats.min(), self._lats.max())
        lon = np.clip(lon, self._lons.min(), self._lons.max())
        alt = np.clip(altitude_m, self._alts.min(), self._alts.max())

        if when is None:
            time = float(self._times[0])
        else:
            origin = dt.datetime(1900, 1, 1)
            seconds = (when - origin).total_seconds()
            time = np.clip(seconds, self._times.min(), self._times.max())

        u_data = self._ds[self.variable_u][:]
        v_data = self._ds[self.variable_v][:]
        if u_data.ndim == 3:
            u_data = u_data[np.newaxis, ...]
            v_data = v_data[np.newaxis, ...]

        u = self._linear_interp(u_data, lat, lon, alt, time)
        v = self._linear_interp(v_data, lat, lon, alt, time)

        if self.variable_w and self.variable_w in self._ds:
            w_data = self._ds[self.variable_w][:]
            if w_data.ndim == 3:
                w_data = w_data[np.newaxis, ...]
            w = self._linear_interp(w_data, lat, lon, alt, time)
        else:
            w = 0.0

        north = v
        east = u
        down = -w
        return north, east, down

    def __del__(self):  # pragma: no cover - destructor safeguard
        self.close()
