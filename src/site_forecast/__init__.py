
import os
os.environ["DARTS_CONFIGURE_MATPLOTLIB"] = "0"

import logging
import configparser
import multiprocessing
from pathlib import Path

import pandas as pd
from astropy import units as u
from astropy.coordinates import Latitude, Longitude

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

KMHOUR_TO_MS = (u.km / u.hour).to("m/s")
TPW_TO_PWV = 1.00016053  # kg/m^2 to mm


class Station:
    def __init__(self: object, name: str, lat: u.Quantity["angle"], lon: u.Quantity["angle"]):
        """
        Parameters
        ----------
        name: str
            Station name. Typically a two-letter code by VLBI convention.
        lat: astropy.units.quantity.Quantity
            Latitude.
        lon: astropy.units.quantity.Quantity
            Longitude.
        """
        self.name = name
        self.latitude  = Latitude(lat)
        self.longitude = Longitude(lon)


SITES = [
        Station("Y1", 34.0773880 * u.deg, -107.6156450 * u.deg),  # VLA
        Station("SC", 17.7565777 * u.deg,  -64.5836305 * u.deg),  # Saint Croix
        Station("HN", 42.9336083 * u.deg,  -71.9865805 * u.deg),  # Hancock
        Station("NL", 41.7714249 * u.deg,  -91.5741333 * u.deg),  # North Liberty
        Station("FD", 30.6350305 * u.deg, -103.9448166 * u.deg),  # Fort Davis
        Station("LA", 35.7751249 * u.deg, -106.2455972 * u.deg),  # Los Alamos
        Station("PT", 34.3010027 * u.deg, -108.1191833 * u.deg),  # Pie Town
        Station("KP", 31.9563055 * u.deg, -111.6124222 * u.deg),  # Kitt Peak
        Station("OV", 37.2316527 * u.deg, -118.2770472 * u.deg),  # Owens Valley
        Station("BR", 48.1312277 * u.deg, -119.6832777 * u.deg),  # Brewster
        Station("MK", 19.8013805 * u.deg, -155.4555027 * u.deg),  # Mauna Kea
        Station("GB", 38.4331222 * u.deg,  -79.8398361 * u.deg),  # Green Bank
]
SITES_BY_NAME = {s.name: s for s in SITES}
SITE_LAT = SITES_BY_NAME["Y1"].latitude
SITE_LON = SITES_BY_NAME["Y1"].longitude


def _load_configuration(filen="site_forecast.ini"):
    config = configparser.ConfigParser()
    config_paths = [
            Path.cwd() / filen,
            Path("~").expanduser() / f".{filen}",
            Path("~/.config").expanduser() / filen
    ]
    for path in config_paths:
        if not path.exists():
            continue
        config.read(path)
        return config
    else:
        raise RuntimeError("No valid configuration file found.")


def _get_logger(config):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    level = config.get("Logging", "log_level", fallback="info").upper()
    log_dir = Path(config.get("Paths", "logs", fallback="./logs")).expanduser()
    today = pd.Timestamp.now().date().isoformat()
    log_path = log_dir / f"forecast_{today}.log"
    if not log_dir.exists():
        log_dir.mkdir()
    try:
        level_code = getattr(logging, level)
    except AttributeError:
        raise RuntimeError(f"Invalid logging level: {level}")
    logging.basicConfig(
            level=level_code,
            format="%(asctime)s\t%(levelname)s\t%(message)s",
            datefmt="%Y-%m-%d %I:%M:%S",
            handlers=[
                    logging.FileHandler(log_path),
                    logging.StreamHandler(),
            ]
    )
    return logging.getLogger()


def _now_dir(t=None):
    if t is None:
        now = pd.Timestamp.now(tz="utc")
    else:
        now = pd.Timestamp(t)
    today = now.date().isoformat().replace("-", "/")
    hour = f"{now.hour:02d}"
    return Path(today) / hour


def run_with_timeout(func, args=None, kwds=None, timeout=120):
    args = args if args is not None else tuple()
    kwds = kwds if kwds is not None else dict()
    if timeout <= 0:
        raise ValueError(f"Timeout must be positive: {timeout=}")
    with multiprocessing.Pool(processes=1) as pool:
        return pool.apply_async(func, args=args, kwds=kwds).get(timeout=timeout)


if __name__ == "__main__":
    # TODO add command line arguments to run
    # - generate a single forecast now
    # - run loop
    # - set the configuration file path
    raise NotImplementedError
else:
    CONFIG = _load_configuration()
    logger = _get_logger(CONFIG)

