"""
Data Loading Module for Exoplanet Transit Detection Pipeline
Handles loading light curve data from various sources including TESS TIC Catalog
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits
import h5py
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, List
import requests
import logging
import json
from urllib.parse import urljoin, quote
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TICCatalogQuerier:
    """
    Query TESS Input Catalog (TIC) from STScI MAST Archive
    Based on: https://archive.stsci.edu/tess/tic_ctl.html
    """
    
    BASE_URL = "https://mast.stsci.edu/api/v0.1/invoke"
    TIC_SEARCH_URL = "https://mast.stsci.edu/api/v0.1/invoke"
    
    def __init__(self):
        """Initialize TIC catalog querier"""
        logger.info("Initializing TIC Catalog Querier")
    
    def query_tic_by_id(self, tic_id: int) -> Dict:
        """
        Query TESS Input Catalog by TIC ID
        
        Args:
            tic_id (int): TESS Input Catalog ID
            
        Returns:
            Dict: TIC catalog entry with all parameters
        """
        try:
            logger.info(f"Querying TIC ID: {tic_id}")
            
            # Construct the request
            request_params = {
                "service": "Mast.Catalogs.Tic.Sql",
                "format": "json",
                "params": {
                    "columns": "*",
                    "filters": [
                        {
                            "paramName": "ID",
                            "operator": "=",
                            "values": [str(tic_id)]
                        }
                    ]
                }
            }
            
            response = requests.post(
                self.TIC_SEARCH_URL,
                data=json.dumps(request_params),
                headers={"Content-type": "application/x-www-form-urlencoded", "Accept": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Query failed with status code {response.status_code}")
            
            data = response.json()
            
            if "data" not in data or len(data["data"]) == 0:
                logger.warning(f"No TIC entry found for ID: {tic_id}")
                return {}
            
            tic_data = data["data"][0]
            logger.info(f"Successfully retrieved TIC ID {tic_id}")
            logger.info(f"Star name: {tic_data.get('dis', 'N/A')}")
            logger.info(f"RA: {tic_data.get('ra', 'N/A')}, Dec: {tic_data.get('dec', 'N/A')}")
            
            return tic_data
            
        except Exception as e:
            logger.error(f"Error querying TIC ID {tic_id}: {str(e)}")
            raise
    
    def query_tic_by_coords(self, ra: float, dec: float, radius: float = 0.1) -> List[Dict]:
        """
        Query TIC by celestial coordinates
        
        Args:
            ra (float): Right Ascension in degrees
            dec (float): Declination in degrees
            radius (float): Search radius in degrees
            
        Returns:
            List[Dict]: List of TIC entries within search radius
        """
        try:
            logger.info(f"Querying TIC by coordinates: RA={ra}, Dec={dec}, Radius={radius}°")
            
            request_params = {
                "service": "Mast.Catalogs.Tic.Sql",
                "format": "json",
                "params": {
                    "columns": "*",
                    "filters": [
                        {
                            "paramName": "ra",
                            "operator": ">",
                            "values": [str(ra - radius)]
                        },
                        {
                            "paramName": "ra",
                            "operator": "<",
                            "values": [str(ra + radius)]
                        },
                        {
                            "paramName": "dec",
                            "operator": ">",
                            "values": [str(dec - radius)]
                        },
                        {
                            "paramName": "dec",
                            "operator": "<",
                            "values": [str(dec + radius)]
                        }
                    ]
                }
            }
            
            response = requests.post(
                self.TIC_SEARCH_URL,
                data=json.dumps(request_params),
                headers={"Content-type": "application/x-www-form-urlencoded", "Accept": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Query failed with status code {response.status_code}")
            
            data = response.json()
            tic_entries = data.get("data", [])
            
            logger.info(f"Found {len(tic_entries)} TIC entries in the search area")
            return tic_entries
            
        except Exception as e:
            logger.error(f"Error querying TIC by coordinates: {str(e)}")
            raise
    
    def query_tic_by_name(self, star_name: str) -> List[Dict]:
        """
        Query TIC by star name
        
        Args:
            star_name (str): Star name or catalog designation
            
        Returns:
            List[Dict]: List of matching TIC entries
        """
        try:
            logger.info(f"Querying TIC by name: {star_name}")
            
            request_params = {
                "service": "Mast.Catalogs.Tic.Sql",
                "format": "json",
                "params": {
                    "columns": "*",
                    "filters": [
                        {
                            "paramName": "dis",
                            "operator": "like",
                            "values": [f"%{star_name}%"]
                        }
                    ]
                }
            }
            
            response = requests.post(
                self.TIC_SEARCH_URL,
                data=json.dumps(request_params),
                headers={"Content-type": "application/x-www-form-urlencoded", "Accept": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Query failed with status code {response.status_code}")
            
            data = response.json()
            tic_entries = data.get("data", [])
            
            logger.info(f"Found {len(tic_entries)} TIC entries matching '{star_name}'")
            return tic_entries
            
        except Exception as e:
            logger.error(f"Error querying TIC by name: {str(e)}")
            raise
    
    def print_tic_info(self, tic_data: Dict):
        """
        Print formatted TIC catalog information
        
        Args:
            tic_data (Dict): TIC catalog entry
        """
        if not tic_data:
            logger.warning("No TIC data to display")
            return
        
        print("\n" + "="*70)
        print("TESS INPUT CATALOG (TIC) ENTRY")
        print("="*70)
        
        key_fields = {
            'ID': 'TIC ID',
            'dis': 'Designation',
            'ra': 'Right Ascension (deg)',
            'dec': 'Declination (deg)',
            'Vmag': 'V Magnitude',
            'Jmag': 'J Magnitude',
            'Hmag': 'H Magnitude',
            'Kmag': 'K Magnitude',
            'Teff': 'Effective Temperature (K)',
            'rad': 'Stellar Radius (R☉)',
            'mass': 'Stellar Mass (M☉)',
            'lum': 'Luminosity (L☉)',
            'd': 'Distance (pc)',
            'plx': 'Parallax (mas)',
        }
        
        for key, label in key_fields.items():
            if key in tic_data:
                value = tic_data[key]
                print(f"{label:.<40} {value}")
        
        print("="*70 + "\n")


class ExoplanetDataLoader:
    """
    A comprehensive data loader for exoplanet light curve data from multiple sources
    with enhanced TESS TIC catalog integration
    """
    
    def __init__(self, cache_dir: str = "./data_cache"):
        """
        Initialize the data loader
        
        Args:
            cache_dir (str): Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.tic_querier = TICCatalogQuerier()
        logger.info(f"Cache directory set to: {self.cache_dir}")
    
    # ==================== GENERAL LOADING METHODS ====================
    
    def load_from_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Auto-detect file format and load data
        
        Args:
            file_path (str): Path to the data file
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: (time, flux, metadata)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        logger.info(f"Loading file: {file_path} (Format: {file_ext})")
        
        if file_ext == '.fits':
            return self.load_fits(str(file_path))
        elif file_ext == '.csv':
            return self.load_csv(str(file_path))
        elif file_ext == '.h5' or file_ext == '.hdf5':
            return self.load_hdf5(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    # ==================== FITS FORMAT ====================
    
    def load_fits(self, file_path: str, hdu_index: int = 1) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load light curve data from FITS file (Kepler/TESS format)
        
        Args:
            file_path (str): Path to FITS file
            hdu_index (int): HDU index containing the light curve data
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: (time, flux, metadata)
        """
        try:
            with fits.open(file_path) as hdul:
                logger.info(f"FITS file opened. Number of HDUs: {len(hdul)}")
                
                # Get primary HDU for metadata
                primary_hdu = hdul[0]
                metadata = dict(primary_hdu.header)
                
                # Get data from specified HDU
                data_hdu = hdul[hdu_index]
                data = data_hdu.data
                
                # Extract time and flux columns
                if 'TIME' in data.dtype.names:
                    time = data['TIME']
                else:
                    raise ValueError("TIME column not found in FITS file")
                
                if 'FLUX' in data.dtype.names:
                    flux = data['FLUX']
                elif 'SAP_FLUX' in data.dtype.names:  # Kepler format
                    flux = data['SAP_FLUX']
                elif 'PDCSAP_FLUX' in data.dtype.names:  # Kepler format (pre-search data conditioning)
                    flux = data['PDCSAP_FLUX']
                elif 'TESSMAG' in data.dtype.names:  # TESS format
                    flux = data['TESSMAG']
                else:
                    raise ValueError("FLUX column not found in FITS file")
                
                # Extract quality flags if available
                if 'SAP_QUALITY' in data.dtype.names:
                    quality = data['SAP_QUALITY']
                    metadata['QUALITY'] = quality
                
                logger.info(f"Loaded {len(time)} data points from FITS file")
                logger.info(f"Time range: {time.min():.2f} - {time.max():.2f}")
                logger.info(f"Flux range: {np.nanmin(flux):.4f} - {np.nanmax(flux):.4f}")
                
                return time, flux, metadata
                
        except Exception as e:
            logger.error(f"Error loading FITS file: {str(e)}")
            raise
    
    # ==================== CSV FORMAT ====================
    
    def load_csv(self, file_path: str, time_col: str = 'time', 
                 flux_col: str = 'flux') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load light curve data from CSV file
        
        Args:
            file_path (str): Path to CSV file
            time_col (str): Name of time column
            flux_col (str): Name of flux column
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: (time, flux, metadata)
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"CSV file loaded with columns: {list(df.columns)}")
            
            # Check if specified columns exist
            if time_col not in df.columns:
                # Try common alternative names
                possible_names = ['time', 'Time', 'TIME', 't', 'jd', 'bjd', 'BJD']
                time_col = next((col for col in possible_names if col in df.columns), None)
                if time_col is None:
                    raise ValueError(f"Time column '{time_col}' not found in CSV")
            
            if flux_col not in df.columns:
                # Try common alternative names
                possible_names = ['flux', 'Flux', 'FLUX', 'magnitude', 'Magnitude', 'mag']
                flux_col = next((col for col in possible_names if col in df.columns), None)
                if flux_col is None:
                    raise ValueError(f"Flux column '{flux_col}' not found in CSV")
            
            time = df[time_col].values.astype(np.float64)
            flux = df[flux_col].values.astype(np.float64)
            
            # Store other columns as metadata
            metadata = {col: df[col].values for col in df.columns if col not in [time_col, flux_col]}
            
            logger.info(f"Loaded {len(time)} data points from CSV file")
            logger.info(f"Time range: {time.min():.2f} - {time.max():.2f}")
            logger.info(f"Flux range: {np.nanmin(flux):.4f} - {np.nanmax(flux):.4f}")
            
            return time, flux, metadata
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    # ==================== HDF5 FORMAT ====================
    
    def load_hdf5(self, file_path: str, time_key: str = 'time', 
                  flux_key: str = 'flux') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load light curve data from HDF5 file
        
        Args:
            file_path (str): Path to HDF5 file
            time_key (str): Key for time dataset
            flux_key (str): Key for flux dataset
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: (time, flux, metadata)
        """
        try:
            with h5py.File(file_path, 'r') as f:
                logger.info(f"HDF5 file opened. Keys: {list(f.keys())}")
                
                # Load time and flux
                if time_key not in f:
                    raise ValueError(f"Time key '{time_key}' not found in HDF5 file")
                if flux_key not in f:
                    raise ValueError(f"Flux key '{flux_key}' not found in HDF5 file")
                
                time = np.array(f[time_key]).astype(np.float64)
                flux = np.array(f[flux_key]).astype(np.float64)
                
                # Extract metadata from attributes
                metadata = {}
                if 'metadata' in f:
                    for key, value in f['metadata'].attrs.items():
                        metadata[key] = value
                
                logger.info(f"Loaded {len(time)} data points from HDF5 file")
                logger.info(f"Time range: {time.min():.2f} - {time.max():.2f}")
                logger.info(f"Flux range: {np.nanmin(flux):.4f} - {np.nanmax(flux):.4f}")
                
                return time, flux, metadata
                
        except Exception as e:
            logger.error(f"Error loading HDF5 file: {str(e)}")
            raise
    
    # ==================== TESS DATA LOADING ====================
    
    def load_tess_data_from_tic(self, tic_id: int, sector: Optional[int] = None,
                               camera: Optional[int] = None, ccd: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Download and load TESS light curve data from a specific TIC ID
        Uses MAST archive API
        
        Args:
            tic_id (int): TESS Input Catalog ID
            sector (Optional[int]): TESS sector (if None, loads all available)
            camera (Optional[int]): TESS camera (1-4)
            ccd (Optional[int]): TESS CCD (1-4)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict]: (time, flux, metadata with TIC info)
        """
        try:
            logger.info(f"Loading TESS data for TIC ID: {tic_id}")
            
            # First, query TIC catalog for this ID
            tic_info = self.tic_querier.query_tic_by_id(tic_id)
            
            if not tic_info:
                logger.warning(f"No TIC information found for ID {tic_id}")
                tic_info = {}
            else:
                self.tic_querier.print_tic_info(tic_info)
            
            # Now query for TESS observations
            logger.info("Querying MAST for TESS observations...")
            
            request_params = {
                "service": "Mast.Caom.Filtered",
                "format": "json",
                "params": {
                    "columns": "*",
                    "filters": [
                        {
                            "paramName": "obs_collection",
                            "operator": "=",
                            "values": ["TESS"]
                        },
                        {
                            "paramName": "dataproduct_type",
                            "operator": "=",
                            "values": ["timeseries"]
                        },
                        {
                            "paramName": "target_name",
                            "operator": "=",
                            "values": [str(tic_id)]
                        }
                    ]
                }
            }
            
            if sector is not None:
                request_params["params"]["filters"].append({
                    "paramName": "obs_id",
                    "operator": "like",
                    "values": [f"tess%-s{sector:04d}-%"]
                })
            
            response = requests.post(
                self.tic_querier.TIC_SEARCH_URL,
                data=json.dumps(request_params),
                headers={"Content-type": "application/x-www-form-urlencoded", "Accept": "application/json"}
            )
            
            if response.status_code != 200:
                raise Exception(f"Query failed with status code {response.status_code}")
            
            observations = response.json().get("data", [])
            logger.info(f"Found {len(observations)} TESS observations for TIC {tic_id}")
            
            if len(observations) == 0:
                raise ValueError(f"No TESS observations found for TIC {tic_id}")
            
            # Download the first available observation (or specified sector)
            obs = observations[0]
            logger.info(f"Downloading observation: {obs.get('obs_id', 'Unknown')}")
            
            # Construct download URL for light curve FITS file
            # TESS light curves are available via MAST
            from astroquery.mast import Observations
            
            obs_table = requests.post(
                self.tic_querier.TIC_SEARCH_URL,
                data=json.dumps(request_params),
                headers={"Content-type": "application/x-www-form-urlencoded", "Accept": "application/json"}
            ).json()
            
            # Get product list
            products_params = {
                "service": "Mast.Caom.Products",
                "format": "json",
                "params": {
                    "columns": "*",
                    "filters": [
                        {
                            "paramName": "obs_collection",
                            "operator": "=",
                            "values": ["TESS"]
                        },
                        {
                            "paramName": "obs_id",
                            "operator": "=",
                            "values": [obs['obs_id']]
                        }
                    ]
                }
            }
            
            products_response = requests.post(
                self.tic_querier.TIC_SEARCH_URL,
                data=json.dumps(products_params),
                headers={"Content-type": "application/x-www-form-urlencoded", "Accept": "application/json"}
            )
            
            products = products_response.json().get("data", [])
            logger.info(f"Found {len(products)} data products")
            
            # Find lc.fits file (light curve)
            lc_file = None
            for product in products:
                if 'lc.fits' in product.get('productFilename', ''):
                    lc_file = product
                    break
            
            if lc_file is None:
                logger.warning("No light curve FITS file found, using first available product")
                if len(products) > 0:
                    lc_file = products[0]
                else:
                    raise ValueError("No data products available for download")
            
            # Construct the download URL
            download_url = lc_file.get('dataProductURL', '')
            if not download_url:
                raise ValueError("No download URL available")
            
            logger.info(f"Downloading from: {download_url}")
            
            # Download the file
            file_name = self.cache_dir / f"tic_{tic_id}_lc.fits"
            response = requests.get(download_url, stream=True)
            
            if response.status_code == 200:
                with open(file_name, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded to: {file_name}")
            else:
                raise Exception(f"Download failed with status code {response.status_code}")
            
            # Load the FITS file
            time, flux, metadata = self.load_fits(str(file_name))
            
            # Add TIC information to metadata
            metadata['TIC_ID'] = tic_id
            metadata['TIC_DATA'] = tic_info
            
            return time, flux, metadata
            
        except ImportError:
            logger.warning("astroquery not installed, trying direct download...")
            raise
        except Exception as e:
            logger.error(f"Error loading TESS data for TIC {tic_id}: {str(e)}")
            raise
    
    def search_tic_by_coords(self, ra: float, dec: float, radius: float = 0.1) -> List[Dict]:
        """
        Search TIC catalog by coordinates and display results
        
        Args:
            ra (float): Right Ascension in degrees
            dec (float): Declination in degrees
            radius (float): Search radius in degrees
            
        Returns:
            List[Dict]: List of TIC entries found
        """
        tic_entries = self.tic_querier.query_tic_by_coords(ra, dec, radius)
        
        if tic_entries:
            print(f"\nFound {len(tic_entries)} TIC entries:")
            print("="*80)
            for i, entry in enumerate(tic_entries[:10], 1):  # Show first 10
                print(f"{i}. TIC ID: {entry.get('ID')}")
                print(f"   Name: {entry.get('dis', 'N/A')}")
                print(f"   RA: {entry.get('ra', 'N/A')}, Dec: {entry.get('dec', 'N/A')}")
                print(f"   Vmag: {entry.get('Vmag', 'N/A')}")
                print(f"   Teff: {entry.get('Teff', 'N/A')} K")
                print()
        
        return tic_entries
    
    def search_tic_by_name(self, star_name: str) -> List[Dict]:
        """
        Search TIC catalog by star name
        
        Args:
            star_name (str): Star name or designation
            
        Returns:
            List[Dict]: List of matching TIC entries
        """
        tic_entries = self.tic_querier.query_tic_by_name(star_name)
        
        if tic_entries:
            print(f"\nFound {len(tic_entries)} TIC entries matching '{star_name}':")
            print("="*80)
            for i, entry in enumerate(tic_entries[:10], 1):
                print(f"{i}. TIC ID: {entry.get('ID')}")
                print(f"   Name: {entry.get('dis', 'N/A')}")
                print(f"   Vmag: {entry.get('Vmag', 'N/A')}")
                print()
        
        return tic_entries
    
    # ==================== BATCH LOADING ====================
    
    def load_batch(self, directory: str, pattern: str = "*.fits") -> Dict[str, Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Load multiple light curve files from a directory
        
        Args:
            directory (str): Directory containing light curve files
            pattern (str): File pattern to match (e.g., "*.fits", "*.csv")
            
        Returns:
            Dict: Dictionary mapping filenames to (time, flux, metadata) tuples
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        files = list(directory.glob(pattern))
        logger.info(f"Found {len(files)} files matching pattern '{pattern}'")
        
        data_dict = {}
        for file_path in files:
            try:
                logger.info(f"Loading {file_path.name}...")
                time, flux, metadata = self.load_from_file(str(file_path))
                data_dict[file_path.name] = (time, flux, metadata)
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(data_dict)} files")
        return data_dict
    
    # ==================== DATA VALIDATION ====================
    
    def validate_data(self, time: np.ndarray, flux: np.ndarray) -> bool:
        """
        Validate loaded light curve data
        
        Args:
            time (np.ndarray): Time array
            flux (np.ndarray): Flux array
            
        Returns:
            bool: True if data is valid
        """
        # Check arrays have same length
        if len(time) != len(flux):
            logger.error(f"Time and flux arrays have different lengths: {len(time)} vs {len(flux)}")
            return False
        
        # Check for minimum data points
        if len(time) < 100:
            logger.warning(f"Very few data points: {len(time)}")
        
        # Check for NaN values
        nan_count = np.sum(np.isnan(flux))
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in flux array")
        
        # Check time is monotonically increasing
        if not np.all(np.diff(time) > 0):
            logger.warning("Time array is not monotonically increasing")
        
        logger.info(f"Data validation passed: {len(time)} points, {nan_count} NaN values")
        return True
    
    # ==================== DATA SUMMARY ====================
    
    def print_data_summary(self, time: np.ndarray, flux: np.ndarray, metadata: Dict):
        """
        Print a summary of loaded data
        
        Args:
            time (np.ndarray): Time array
            flux (np.ndarray): Flux array
            metadata (Dict): Metadata dictionary
        """
        print("\n" + "="*70)
        print("LIGHT CURVE DATA SUMMARY")
        print("="*70)
        print(f"Number of data points: {len(time)}")
        print(f"Time range: {time.min():.4f} - {time.max():.4f} days")
        print(f"Time span: {time.max() - time.min():.4f} days")
        print(f"Time resolution: {np.mean(np.diff(time))*24*60:.4f} minutes")
        print(f"\nFlux range: {np.nanmin(flux):.6f} - {np.nanmax(flux):.6f}")
        print(f"Flux mean: {np.nanmean(flux):.6f}")
        print(f"Flux std: {np.nanstd(flux):.6f}")
        print(f"NaN values in flux: {np.sum(np.isnan(flux))}")
        
        if 'TIC_ID' in metadata:
            print(f"\nTIC ID: {metadata['TIC_ID']}")
        
        print(f"\nTotal metadata fields: {len(metadata)}")
        
        if 'TIC_DATA' in metadata and metadata['TIC_DATA']:
            tic_data = metadata['TIC_DATA']
            print("\nTIC Catalog Information:")
            print(f"  Designation: {tic_data.get('dis', 'N/A')}")
            print(f"  RA/Dec: {tic_data.get('ra', 'N/A')} / {tic_data.get('dec', 'N/A')}")
            print(f"  V Magnitude: {tic_data.get('Vmag', 'N/A')}")
            print(f"  Effective Temperature: {tic_data.get('Teff', 'N/A')} K")
            print(f"  Stellar Radius: {tic_data.get('rad', 'N/A')} R☉")
            print(f"  Distance: {tic_data.get('d', 'N/A')} pc")
        
        print("="*70 + "\n")


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    
    # Create data loader instance
    loader = ExoplanetDataLoader(cache_dir="./exoplanet_data")
    
    print("\n" + "="*70)
    print("TESS/EXOPLANET DATA LOADER - EXAMPLES")
    print("="*70)
    
    # Example 1: Load from a local FITS file
    print("\n--- Example 1: Load from local FITS file ---")
    print("Uncomment to use:")
    print("time, flux, metadata = loader.load_fits('./sample_data/kepler_light_curve.fits')")
    print("loader.validate_data(time, flux)")
    print("loader.print_data_summary(time, flux, metadata)")
    
    # Example 2: Load from CSV file
    print("\n--- Example 2: Load from CSV file ---")
    print("Uncomment to use:")
    print("time, flux, metadata = loader.load_csv('./sample_data/light_curve.csv')")
    print("loader.validate_data(time, flux)")
    print("loader.print_data_summary(time, flux, metadata)")
    
    # Example 3: Search TIC by coordinates
    print("\n--- Example 3: Search TIC Catalog by Coordinates ---")
    print("Example: loader.search_tic_by_coords(ra=290.5, dec=-62.5, radius=0.5)")
    print("Returns list of TIC entries in that region")
    
    # Example 4: Search TIC by name
    print("\n--- Example 4: Search TIC Catalog by Star Name ---")
    print("Example: results = loader.search_tic_by_name('TIC 25155310')")
    print("Returns matching TIC entries")
    
    # Example 5: Load TESS data by TIC ID (requires internet)
    print("\n--- Example 5: Load TESS Data by TIC ID ---")
    print("Uncomment to use (requires internet and MAST access):")
    print("time, flux, metadata = loader.load_tess_data_from_tic(tic_id=25155310)")
    print("loader.validate_data(time, flux)")
    print("loader.print_data_summary(time, flux, metadata)")
    
    # Example 6: Batch load files
    print("\n--- Example 6: Batch Load Files ---")
    print("Uncomment to use:")
    print("data_dict = loader.load_batch('./sample_data/', pattern='*.fits')")
    print("for filename, (time, flux, metadata) in data_dict.items():")
    print("    print(f'{filename}: {len(time)} data points')")
    
    print("\n" + "="*70)
    print("Data loader module initialized successfully!")
    print("="*70 + "\n")
