# -*- coding: utf-8 -*-
"""Download LULC rasters from Google Earth Engine for a set of regions.

Three data sources are supported:

**MapBiomas** (``--source mapbiomas``):
    Annual land-use/land-cover classification for South America and other
    regions.  Default collection is Brazil (Collection 8).  Pass
    ``--mapbiomas-collection`` to switch to another region, or use
    ``--mapbiomas-collection-id`` to supply a full GEE asset path.

**Dynamic World** (``--source dynamic-world``):
    Google / World Resources Institute near-real-time global LULC at 10 m
    resolution.  The modal label band is computed over the selected calendar
    year.

**ESRI LULC** (``--source esri-lulc``):
    ESRI annual global LULC at 10 m resolution (Sentinel-2).

Two download modes are supported:

**direct** (default):
    Downloads images via ``getDownloadURL``.  Works reliably for regions up to
    roughly 1° × 1° at 10–30 m resolution.  No Google Drive or Cloud Storage
    account required.

**export-drive**:
    Submits a ``Export.image.toDrive`` task and polls until it completes.
    Suitable for large regions.  The output file lands in Google Drive; a
    manual download step is still needed unless Drive is mounted locally.

Authentication
--------------
Google Earth Engine does **not** support plain username / password
authentication.  Two mechanisms are available:

*OAuth2 (interactive)* — default when no service account is configured.
    Calls ``ee.Authenticate()``, which opens a browser window (or prints a URL
    for code-based flow if ``auth_mode="notebook"`` is used).

*Service account* — non-interactive, suitable for CI/scripts.
    Provide ``--service-account-email`` and ``--service-account-key-file``
    (a JSON private-key file downloaded from the Google Cloud Console).
"""

import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import requests
from shapely.geometry import box, mapping

try:
    import ee
except ImportError:  # pragma: no cover
    ee = None  # type: ignore[assignment]

try:
    import httplib2
except ImportError:  # pragma: no cover
    httplib2 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MapBiomas collection registry
# ---------------------------------------------------------------------------

MAPBIOMAS_COLLECTIONS: Dict[str, str] = {
    "brazil": (
        "projects/mapbiomas-workspace/public/collection8/"
        "mapbiomas_collection80_integration_v1"
    ),
    "amazon": (
        "projects/mapbiomas-raisg/public/collection5/"
        "mapbiomas_raisg_panamazonia_collection5_integration_v1"
    ),
    "chaco": (
        "projects/mapbiomas-chaco/public/collection4/"
        "mapbiomas_chaco_collection4_integration_v1"
    ),
    "atlantic_forest": (
        "projects/mapbiomas-af/public/collection3/"
        "mapbiomas_atlanticforest_collection3_integration_v1"
    ),
    "pampa": (
        "projects/mapbiomas-pampa/public/collection3/"
        "mapbiomas_pampa_collection3_integration_v1"
    ),
    "peru": (
        "projects/ibc-mapbiomas/public/collection1/"
        "mapbiomas_peru_collection1_integration_v1"
    ),
    "bolivia": (
        "projects/mapbiomas-bolivia/public/collection1/"
        "mapbiomas_bolivia_collection1_integration_v1"
    ),
    "colombia": (
        "projects/mapbiomas-colombia/public/collection1/"
        "mapbiomas_colombia_collection1_integration_v1"
    ),
    "venezuela": (
        "projects/mapbiomas-venezuela/public/collection1/"
        "mapbiomas_venezuela_collection1_integration_v1"
    ),
    "ecuador": (
        "projects/mapbiomas-ecuador/public/collection1/"
        "mapbiomas_ecuador_collection1_integration_v1"
    ),
    "paraguay": (
        "projects/mapbiomas-paraguay/public/collection1/"
        "mapbiomas_paraguay_collection1_integration_v1"
    ),
    "argentina": (
        "projects/mapbiomas-agriculture/public/collection1/"
        "mapbiomas_argentina_collection1_integration_v1"
    ),
    "uruguay": (
        "projects/mapbiomas-uruguay/public/collection1/"
        "mapbiomas_uruguay_collection1_integration_v1"
    ),
    "indonesia": (
        "projects/mapbiomas-indonesia/public/collection2/"
        "mapbiomas_indonesia_collection2_integration_v1"
    ),
}

DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
ESRI_LULC_COLLECTION = "projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS"

_DEFAULT_SCALES: Dict[str, int] = {
    "mapbiomas": 30,
    "dynamic-world": 10,
    "esri-lulc": 10,
}

_VALID_SOURCES = {"mapbiomas", "dynamic-world", "esri-lulc"}


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def _require_ee() -> None:
    """Raise ImportError if earthengine-api is not installed."""
    if ee is None:
        raise ImportError(
            "The 'earthengine-api' package is required for GEE downloads. "
            "Install it with: pip install earthengine-api"
        )


def _build_http_transport(proxy_url: str, ca_bundle: Optional[str] = None):
    """Build an ``httplib2.Http`` transport configured with a proxy.

    ``httplib2`` is a dependency of ``earthengine-api`` and is always
    available when ``ee`` is installed.

    Args:
        proxy_url: Proxy URL including scheme, optional credentials, host, and
            port.  Examples: ``http://proxy.corp:8080``,
            ``http://user:pass@proxy.corp:8080``.
        ca_bundle: Path to a CA certificate bundle (``.crt`` / ``.pem``) for
            SSL verification.  ``None`` lets ``httplib2`` use its defaults.

    Returns:
        Configured ``httplib2.Http`` instance suitable for passing as
        ``http_transport`` to ``ee.Initialize()``.
    """
    from urllib.parse import urlparse

    parsed = urlparse(proxy_url)
    proxy_info = httplib2.ProxyInfo(
        proxy_type=httplib2.socks.PROXY_TYPE_HTTP,
        proxy_host=parsed.hostname,
        proxy_port=parsed.port or 8080,
        proxy_user=parsed.username,
        proxy_pass=parsed.password,
    )
    return httplib2.Http(proxy_info=proxy_info, ca_certs=ca_bundle)


def initialize_gee(
    project: Optional[str] = None,
    service_account_email: Optional[str] = None,
    service_account_key_file: Optional[str] = None,
    proxy_url: Optional[str] = None,
    ca_bundle: Optional[str] = None,
) -> None:
    """Initialize Google Earth Engine.

    Uses service account credentials when both ``service_account_email`` and
    ``service_account_key_file`` are provided.  Otherwise triggers the
    interactive OAuth2 flow via ``ee.Authenticate()``.

    Note: GEE does not support plain username/password authentication.
    For non-interactive / server use, provide a service account JSON key.
    For interactive use, ``ee.Authenticate()`` opens a browser or prints an
    authorization URL.

    Args:
        project: GEE Cloud project ID.
        service_account_email: Service account email (e.g.
            ``mybot@my-project.iam.gserviceaccount.com``).
        service_account_key_file: Path to the service account private key JSON
            file downloaded from the Google Cloud Console.
        proxy_url: HTTP proxy URL (e.g. ``http://user:pass@proxy.corp:8080``).
            When set, an ``httplib2.Http`` transport is built and injected into
            ``ee.Initialize()``.
        ca_bundle: Path to a CA certificate bundle for SSL verification.
            Used together with ``proxy_url`` when the proxy uses a self-signed
            certificate.

    Raises:
        ImportError: If ``earthengine-api`` is not installed.

    Example YAML:
        ```yaml
        gee_auth:
          project: my-gee-project
          service_account_email: mybot@my-project.iam.gserviceaccount.com
          service_account_key_file: /secrets/gee_key.json
          proxy_url: http://proxy.corp:8080
          ca_bundle: /certs/corp-ca.crt
        ```
    """
    _require_ee()
    init_kwargs: dict = {"project": project}
    if proxy_url:
        init_kwargs["http_transport"] = _build_http_transport(proxy_url, ca_bundle)

    if service_account_email and service_account_key_file:
        init_kwargs["credentials"] = ee.ServiceAccountCredentials(
            service_account_email, service_account_key_file
        )
        ee.Initialize(**init_kwargs)
    else:
        ee.Authenticate()
        ee.Initialize(**init_kwargs)


# ---------------------------------------------------------------------------
# Region loading
# ---------------------------------------------------------------------------


def load_regions_from_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> List[Tuple[str, dict]]:
    """Build a single region from a bounding box in WGS84.

    Args:
        xmin: Minimum longitude (west), in decimal degrees.
        ymin: Minimum latitude (south), in decimal degrees.
        xmax: Maximum longitude (east), in decimal degrees.
        ymax: Maximum latitude (north), in decimal degrees.

    Returns:
        List with one element ``[("bbox", geojson_geometry)]`` where
        ``geojson_geometry`` is the GeoJSON dict of the rectangle.

    Example:
        >>> load_regions_from_bbox(-48.0, -23.5, -47.5, -23.0)
        [('bbox', {'type': 'Polygon', 'coordinates': [...]})]
    """
    geom = box(xmin, ymin, xmax, ymax)
    return [("bbox", mapping(geom))]


def load_regions_from_vector(
    vector_path: str,
    name_attribute: Optional[str] = None,
) -> List[Tuple[str, dict]]:
    """Load regions from a GeoPackage or GeoJSON vector file.

    Each feature becomes a separate region.  Non-geographic CRS inputs are
    reprojected to EPSG:4326 automatically.

    Args:
        vector_path: Path to a GeoPackage (``.gpkg``) or GeoJSON file.
        name_attribute: Column name whose value is used to name each output
            file.  When ``None`` or missing from the file, falls back to the
            zero-based row index as ``"region_0"``, ``"region_1"``, …

    Returns:
        List of ``(name, geojson_geometry)`` tuples, one per feature.

    Raises:
        ValueError: If ``geopandas`` cannot read the file.

    Example YAML:
        ```yaml
        gee_download:
          vector_path: /data/municipalities.gpkg
          name_attribute: cod_ibge
        ```
    """
    gdf = gpd.read_file(vector_path)
    if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")

    regions: List[Tuple[str, dict]] = []
    for idx, row in gdf.iterrows():
        if name_attribute and name_attribute in gdf.columns:
            name = str(row[name_attribute])
        else:
            name = f"region_{idx}"
        regions.append((name, mapping(row.geometry)))
    return regions


def load_regions(
    bbox: Optional[Tuple[float, float, float, float]] = None,
    vector_path: Optional[str] = None,
    name_attribute: Optional[str] = None,
) -> List[Tuple[str, dict]]:
    """Load regions from either a bounding box or a vector file.

    Exactly one of ``bbox`` or ``vector_path`` must be provided.

    Args:
        bbox: ``(xmin, ymin, xmax, ymax)`` in WGS84 decimal degrees.
        vector_path: Path to a GeoPackage or GeoJSON file.
        name_attribute: Polygon attribute used for output file naming
            (vector_path mode only).

    Returns:
        List of ``(name, geojson_geometry)`` tuples.

    Raises:
        ValueError: If neither or both of ``bbox`` and ``vector_path`` are
            given.
    """
    if bbox is None and vector_path is None:
        raise ValueError("Either 'bbox' or 'vector_path' must be provided.")
    if bbox is not None and vector_path is not None:
        raise ValueError("Only one of 'bbox' or 'vector_path' may be provided.")
    if bbox is not None:
        return load_regions_from_bbox(*bbox)
    return load_regions_from_vector(vector_path, name_attribute)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# GEE image builders
# ---------------------------------------------------------------------------


def _ee_geometry_from_geojson(geojson: dict):
    """Convert a GeoJSON geometry dict to an ``ee.Geometry``.

    Args:
        geojson: GeoJSON geometry dict with ``'type'`` and ``'coordinates'``
            keys.

    Returns:
        ``ee.Geometry`` instance.
    """
    return ee.Geometry(geojson)


def _get_mapbiomas_image(year: int, collection_id: str):
    """Return the single-band MapBiomas classification image for a year.

    Args:
        year: Classification year (e.g. ``2022``).
        collection_id: Full GEE asset ID for the MapBiomas collection.

    Returns:
        ``ee.Image`` with the band ``classification_{year}``.
    """
    return ee.Image(collection_id).select(f"classification_{year}")


def _get_dynamic_world_image(region_geom, year: int):
    """Return the modal Dynamic World label image for a year and region.

    Args:
        region_geom: ``ee.Geometry`` defining the region of interest.
        year: Calendar year to filter the collection.

    Returns:
        ``ee.Image`` with a single ``label`` band (most frequent class).
    """
    return (
        ee.ImageCollection(DYNAMIC_WORLD_COLLECTION)
        .filterBounds(region_geom)
        .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
        .select("label")
        .mode()
    )


def _get_esri_lulc_image(region_geom, year: int):
    """Return the ESRI LULC image for a year and region.

    Args:
        region_geom: ``ee.Geometry`` defining the region of interest.
        year: Year to select from the ESRI LULC time-series collection.

    Returns:
        ``ee.Image`` with a single ``b1`` band.
    """
    return (
        ee.ImageCollection(ESRI_LULC_COLLECTION)
        .filterBounds(region_geom)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .first()
        .select("b1")
    )


# ---------------------------------------------------------------------------
# Download backends
# ---------------------------------------------------------------------------


def _download_image_direct(
    image,
    region_geom,
    output_path: Path,
    scale: int,
    crs: str,
    proxy_url: Optional[str] = None,
    ca_bundle: Optional[str] = None,
) -> None:
    """Download a GEE image to a GeoTIFF via ``getDownloadURL``.

    The response may be a bare GeoTIFF or a ZIP archive; both are handled.

    Args:
        image: ``ee.Image`` to download.
        region_geom: ``ee.Geometry`` defining the clip/download extent.
        output_path: Destination GeoTIFF path.
        scale: Pixel resolution in metres.
        crs: Output CRS string (e.g. ``"EPSG:4326"``).
        proxy_url: Optional HTTP proxy URL.  When set, passed as the
            ``proxies`` dict to ``requests.get()``.
        ca_bundle: Optional path to a CA certificate bundle for SSL
            verification.  When set, passed as ``verify`` to
            ``requests.get()``.

    Raises:
        requests.HTTPError: If the HTTP request fails.
        ValueError: If the ZIP response contains no ``.tif`` file.
    """
    url = image.getDownloadURL(
        {
            "scale": scale,
            "crs": crs,
            "region": region_geom,
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )
    request_kwargs: dict = {"stream": True, "timeout": 300}
    if proxy_url:
        request_kwargs["proxies"] = {"http": proxy_url, "https": proxy_url}
    if ca_bundle:
        request_kwargs["verify"] = ca_bundle
    response = requests.get(url, **request_kwargs)
    response.raise_for_status()

    content = BytesIO(response.content)
    if zipfile.is_zipfile(content):
        content.seek(0)
        with zipfile.ZipFile(content) as zf:
            tif_names = [n for n in zf.namelist() if n.endswith(".tif")]
            if not tif_names:
                raise ValueError(
                    f"No .tif file found in download zip for {output_path.name}"
                )
            with zf.open(tif_names[0]) as src, open(output_path, "wb") as dst:
                dst.write(src.read())
    else:
        content.seek(0)
        with open(output_path, "wb") as f:
            f.write(content.read())


def _export_image_to_drive(
    image,
    region_geom,
    file_name: str,
    drive_folder: str,
    scale: int,
    crs: str,
    poll_interval: int = 10,
) -> str:
    """Submit a GEE export-to-Drive task and wait for completion.

    The output GeoTIFF lands in the specified Google Drive folder.  A manual
    download from Drive is required unless Drive is mounted locally.

    Args:
        image: ``ee.Image`` to export.
        region_geom: ``ee.Geometry`` defining the export extent.
        file_name: Base filename (without extension) for the Drive file.
        drive_folder: Google Drive folder name where the file will be saved.
        scale: Pixel resolution in metres.
        crs: Output CRS string.
        poll_interval: Seconds between task-status polls.

    Returns:
        Final task state string (``"COMPLETED"``).

    Raises:
        RuntimeError: If the export task ends in a non-COMPLETED state.
    """
    task = ee.batch.Export.image.toDrive(
        image=image.clip(region_geom),
        description=file_name,
        folder=drive_folder,
        fileNamePrefix=file_name,
        scale=scale,
        crs=crs,
        region=region_geom,
        maxPixels=int(1e13),
        fileFormat="GeoTIFF",
    )
    task.start()
    logger.info(
        "Export task started for '%s'. Polling every %ds…", file_name, poll_interval
    )

    while task.active():
        time.sleep(poll_interval)

    status = task.status()
    state = status["state"]
    if state != "COMPLETED":
        raise RuntimeError(
            f"GEE export task for '{file_name}' ended with state '{state}': "
            f"{status.get('error_message', 'no error message')}"
        )
    logger.info("Export task completed for '%s'.", file_name)
    return state


# ---------------------------------------------------------------------------
# Per-source download wrappers
# ---------------------------------------------------------------------------


def download_mapbiomas(
    name: str,
    region_geojson: dict,
    output_path: Path,
    year: int,
    collection: str = "brazil",
    collection_id: Optional[str] = None,
    download_mode: str = "direct",
    scale: int = 30,
    crs: str = "EPSG:4326",
    drive_folder: str = "GEE_Downloads",
    proxy_url: Optional[str] = None,
    ca_bundle: Optional[str] = None,
) -> None:
    """Download a MapBiomas classification raster for one region.

    Args:
        name: Region identifier used in log messages.
        region_geojson: GeoJSON geometry dict of the region.
        output_path: Destination GeoTIFF path.
        year: Classification year (e.g. ``2022``).
        collection: MapBiomas region key, one of the keys in
            :data:`MAPBIOMAS_COLLECTIONS`.  Ignored when ``collection_id``
            is set.
        collection_id: Full GEE asset ID override.  When ``None``, the ID is
            looked up from ``MAPBIOMAS_COLLECTIONS[collection]``.
        download_mode: ``"direct"`` or ``"export-drive"``.
        scale: Pixel resolution in metres (default 30 m).
        crs: Output CRS.
        drive_folder: Drive folder name (export-drive mode only).
        proxy_url: Optional HTTP proxy URL for the download request.
        ca_bundle: Optional CA certificate bundle path for SSL verification.

    Raises:
        KeyError: If ``collection`` is not in :data:`MAPBIOMAS_COLLECTIONS`.
        ImportError: If ``earthengine-api`` is not installed.

    Example YAML:
        ```yaml
        gee_download:
          source: mapbiomas
          year: 2022
          collection: brazil
          bbox: [-48.5, -23.5, -47.5, -22.5]
          output_dir: /data/mapbiomas
        ```
    """
    _require_ee()
    if collection_id is None:
        if collection not in MAPBIOMAS_COLLECTIONS:
            raise KeyError(
                f"Unknown MapBiomas collection '{collection}'. "
                f"Valid options: {sorted(MAPBIOMAS_COLLECTIONS)}"
            )
        collection_id = MAPBIOMAS_COLLECTIONS[collection]

    region_geom = _ee_geometry_from_geojson(region_geojson)
    image = _get_mapbiomas_image(year, collection_id)

    if download_mode == "direct":
        _download_image_direct(
            image,
            region_geom,
            output_path,
            scale,
            crs,
            proxy_url=proxy_url,
            ca_bundle=ca_bundle,
        )
    else:
        _export_image_to_drive(
            image, region_geom, output_path.stem, drive_folder, scale, crs
        )


def download_dynamic_world(
    name: str,
    region_geojson: dict,
    output_path: Path,
    year: int,
    download_mode: str = "direct",
    scale: int = 10,
    crs: str = "EPSG:4326",
    drive_folder: str = "GEE_Downloads",
    proxy_url: Optional[str] = None,
    ca_bundle: Optional[str] = None,
) -> None:
    """Download a Dynamic World modal-label raster for one region.

    Args:
        name: Region identifier used in log messages.
        region_geojson: GeoJSON geometry dict of the region.
        output_path: Destination GeoTIFF path.
        year: Year to filter the Dynamic World collection.
        download_mode: ``"direct"`` or ``"export-drive"``.
        scale: Pixel resolution in metres (default 10 m, DW native).
        crs: Output CRS.
        drive_folder: Drive folder name (export-drive mode only).
        proxy_url: Optional HTTP proxy URL for the download request.
        ca_bundle: Optional CA certificate bundle path for SSL verification.

    Raises:
        ImportError: If ``earthengine-api`` is not installed.

    Example YAML:
        ```yaml
        gee_download:
          source: dynamic-world
          year: 2022
          bbox: [-48.5, -23.5, -47.5, -22.5]
          output_dir: /data/dynamic_world
        ```
    """
    _require_ee()
    region_geom = _ee_geometry_from_geojson(region_geojson)
    image = _get_dynamic_world_image(region_geom, year)

    if download_mode == "direct":
        _download_image_direct(
            image,
            region_geom,
            output_path,
            scale,
            crs,
            proxy_url=proxy_url,
            ca_bundle=ca_bundle,
        )
    else:
        _export_image_to_drive(
            image, region_geom, output_path.stem, drive_folder, scale, crs
        )


def download_esri_lulc(
    name: str,
    region_geojson: dict,
    output_path: Path,
    year: int,
    download_mode: str = "direct",
    scale: int = 10,
    crs: str = "EPSG:4326",
    drive_folder: str = "GEE_Downloads",
    proxy_url: Optional[str] = None,
    ca_bundle: Optional[str] = None,
) -> None:
    """Download an ESRI LULC raster for one region.

    Args:
        name: Region identifier used in log messages.
        region_geojson: GeoJSON geometry dict of the region.
        output_path: Destination GeoTIFF path.
        year: Year to select from the ESRI LULC time-series.
        download_mode: ``"direct"`` or ``"export-drive"``.
        scale: Pixel resolution in metres (default 10 m, ESRI native).
        crs: Output CRS.
        drive_folder: Drive folder name (export-drive mode only).
        proxy_url: Optional HTTP proxy URL for the download request.
        ca_bundle: Optional CA certificate bundle path for SSL verification.

    Raises:
        ImportError: If ``earthengine-api`` is not installed.

    Example YAML:
        ```yaml
        gee_download:
          source: esri-lulc
          year: 2022
          bbox: [-48.5, -23.5, -47.5, -22.5]
          output_dir: /data/esri_lulc
        ```
    """
    _require_ee()
    region_geom = _ee_geometry_from_geojson(region_geojson)
    image = _get_esri_lulc_image(region_geom, year)

    if download_mode == "direct":
        _download_image_direct(
            image,
            region_geom,
            output_path,
            scale,
            crs,
            proxy_url=proxy_url,
            ca_bundle=ca_bundle,
        )
    else:
        _export_image_to_drive(
            image, region_geom, output_path.stem, drive_folder, scale, crs
        )


# ---------------------------------------------------------------------------
# Thread worker
# ---------------------------------------------------------------------------


def _download_one_region(
    name: str,
    region_geojson: dict,
    source: str,
    output_dir: Path,
    year: int,
    download_mode: str,
    scale: int,
    crs: str,
    drive_folder: str,
    mapbiomas_collection: str,
    mapbiomas_collection_id: Optional[str],
    proxy_url: Optional[str] = None,
    ca_bundle: Optional[str] = None,
) -> Tuple[str, bool, Optional[str]]:
    """Download one region and return ``(name, success, error_message)``.

    Skips the download if the output file already exists.
    """
    output_path = output_dir / f"{name}.tif"
    if output_path.exists():
        logger.info("Skipping '%s' — already exists at %s.", name, output_path)
        return name, True, None

    try:
        logger.info("Downloading '%s' from %s for year %d…", name, source, year)
        common_kwargs: dict = dict(
            name=name,
            region_geojson=region_geojson,
            output_path=output_path,
            year=year,
            download_mode=download_mode,
            scale=scale,
            crs=crs,
            drive_folder=drive_folder,
            proxy_url=proxy_url,
            ca_bundle=ca_bundle,
        )
        if source == "mapbiomas":
            download_mapbiomas(
                **common_kwargs,
                collection=mapbiomas_collection,
                collection_id=mapbiomas_collection_id,
            )
        elif source == "dynamic-world":
            download_dynamic_world(**common_kwargs)
        else:
            download_esri_lulc(**common_kwargs)
        logger.info("Done: '%s' → %s.", name, output_path)
        return name, True, None
    except Exception as exc:
        logger.error("Failed to download region '%s': %s", name, exc)
        return name, False, str(exc)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def download_regions(
    regions: List[Tuple[str, dict]],
    output_dir: Path,
    source: str,
    year: int,
    download_mode: str = "direct",
    max_workers: int = 4,
    scale: Optional[int] = None,
    crs: str = "EPSG:4326",
    drive_folder: str = "GEE_Downloads",
    mapbiomas_collection: str = "brazil",
    mapbiomas_collection_id: Optional[str] = None,
    proxy_url: Optional[str] = None,
    ca_bundle: Optional[str] = None,
) -> Dict[str, bool]:
    """Download LULC rasters for all regions in parallel.

    Uses a ``ThreadPoolExecutor`` with up to ``max_workers`` threads.
    Regions whose output file already exists are skipped without re-downloading.

    Args:
        regions: List of ``(name, geojson_geometry)`` tuples from
            :func:`load_regions`.
        output_dir: Directory where output GeoTIFFs will be written.  Created
            if it does not exist.
        source: One of ``"mapbiomas"``, ``"dynamic-world"``, ``"esri-lulc"``.
        year: Target year for the LULC product.
        download_mode: ``"direct"`` (getDownloadURL) or ``"export-drive"``
            (Google Drive export).
        max_workers: Maximum parallel download threads.
        scale: Pixel resolution in metres.  Defaults to the product native
            resolution (30 m for MapBiomas, 10 m for Dynamic World / ESRI).
        crs: Output coordinate reference system.
        drive_folder: Google Drive folder name (export-drive mode only).
        mapbiomas_collection: MapBiomas region key (default ``"brazil"``).
            Ignored for other sources.
        mapbiomas_collection_id: Full GEE asset ID override for MapBiomas.
            When set, takes precedence over ``mapbiomas_collection``.
        proxy_url: Optional HTTP proxy URL applied to every download thread.
        ca_bundle: Optional CA certificate bundle path for SSL verification.

    Returns:
        Dict mapping each region name to a success flag (``True`` / ``False``).

    Raises:
        ValueError: If ``source`` is not one of the valid options.

    Example YAML:
        ```yaml
        gee_download:
          source: mapbiomas
          year: 2022
          vector_path: /data/municipalities.gpkg
          name_attribute: cod_ibge
          output_dir: /data/mapbiomas
          max_workers: 8
          download_mode: direct
        ```
    """
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"Unknown source '{source}'. Valid options: {sorted(_VALID_SOURCES)}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_scale = scale if scale is not None else _DEFAULT_SCALES[source]

    results: Dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _download_one_region,
                name,
                region_geojson,
                source,
                output_dir,
                year,
                download_mode,
                effective_scale,
                crs,
                drive_folder,
                mapbiomas_collection,
                mapbiomas_collection_id,
                proxy_url,
                ca_bundle,
            ): name
            for name, region_geojson in regions
        }
        for future in as_completed(futures):
            name, success, _ = future.result()
            results[name] = success

    return results
