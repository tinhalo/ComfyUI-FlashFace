import os
import hashlib
import requests
from tqdm import tqdm
import logging
import shutil

logger = logging.getLogger("flashface.cache_utils")

def calculate_md5(file_path):
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def download_file(url, dest_path, expected_md5=None):
    """
    Download a file with progress bar and MD5 verification.
    
    Args:
        url (str): URL to download from
        dest_path (str): Destination path to save the file
        expected_md5 (str, optional): Expected MD5 hash for verification
        
    Returns:
        bool: True if download was successful and MD5 matches (if provided)
    """
    try:
        import pydash as _
        
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Don't download if file exists and MD5 matches
        if os.path.exists(dest_path) and expected_md5:
            existing_md5 = calculate_md5(dest_path)
            if existing_md5 == expected_md5:
                logger.info(f"File already exists with correct MD5: {dest_path}")
                return True
            else:
                logger.warning(f"File exists but MD5 doesn't match. Re-downloading: {dest_path}")
                
        # Create a temporary file for downloading
        temp_path = f"{dest_path}.download"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            file_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(temp_path, 'wb') as f, tqdm(
                    desc=os.path.basename(dest_path),
                    total=file_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
            
            # Verify MD5 if provided
            if expected_md5:
                downloaded_md5 = calculate_md5(temp_path)
                if downloaded_md5 != expected_md5:
                    logger.error(f"MD5 verification failed: {downloaded_md5} != {expected_md5}")
                    _.attempt(lambda: os.remove(temp_path))
                    return False
            
            # Move temporary file to final destination
            shutil.move(temp_path, dest_path)
            logger.info(f"Successfully downloaded: {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            _.attempt(lambda: os.remove(temp_path))
            return False
            
    except ImportError:
        # Fallback to original implementation
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Don't download if file exists and MD5 matches
        if os.path.exists(dest_path) and expected_md5:
            existing_md5 = calculate_md5(dest_path)
            if existing_md5 == expected_md5:
                logger.info(f"File already exists with correct MD5: {dest_path}")
                return True
            else:
                logger.warning(f"File exists but MD5 doesn't match. Re-downloading: {dest_path}")
                
        # Create a temporary file for downloading
        temp_path = f"{dest_path}.download"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            file_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(temp_path, 'wb') as f, tqdm(
                    desc=os.path.basename(dest_path),
                    total=file_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
            
            # Verify MD5 if provided
            if expected_md5:
                downloaded_md5 = calculate_md5(temp_path)
                if downloaded_md5 != expected_md5:
                    logger.error(f"MD5 verification failed: {downloaded_md5} != {expected_md5}")
                    os.remove(temp_path)
                    return False
            
            # Move temporary file to final destination
            shutil.move(temp_path, dest_path)
            logger.info(f"Successfully downloaded: {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

def ensure_file_exists(url, dest_path, expected_md5=None, description=None):
    """
    Ensure a file exists, downloading it if necessary.
    
    Args:
        url (str): URL to download from if file doesn't exist
        dest_path (str): Path where file should be located
        expected_md5 (str, optional): Expected MD5 hash for verification
        description (str, optional): Description of the file for logging
        
    Returns:
        str: Path to the file if it exists or was successfully downloaded
        None: If file doesn't exist and couldn't be downloaded
    """
    file_desc = description or os.path.basename(dest_path)
    
    try:
        import pydash as _
        
        if os.path.exists(dest_path):
            if expected_md5:
                actual_md5 = calculate_md5(dest_path)
                if actual_md5 != expected_md5:
                    logger.warning(f"{file_desc} exists but has incorrect MD5. Expected: {expected_md5}, Got: {actual_md5}")
                    logger.info(f"Re-downloading {file_desc}...")
                    return _.cond([
                        [lambda: download_file(url, dest_path, expected_md5), lambda: dest_path],
                        [_.constant(True), lambda: None]
                    ])()
                else:
                    logger.info(f"{file_desc} exists with correct MD5")
            else:
                logger.info(f"{file_desc} exists (MD5 not verified)")
            return dest_path
        else:
            logger.info(f"{file_desc} not found, downloading...")
            return _.cond([
                [lambda: download_file(url, dest_path, expected_md5), lambda: dest_path],
                [_.constant(True), lambda: None]
            ])()
            
    except ImportError:
        # Fallback to original implementation
        if os.path.exists(dest_path):
            if expected_md5:
                actual_md5 = calculate_md5(dest_path)
                if actual_md5 != expected_md5:
                    logger.warning(f"{file_desc} exists but has incorrect MD5. Expected: {expected_md5}, Got: {actual_md5}")
                    logger.info(f"Re-downloading {file_desc}...")
                    if not download_file(url, dest_path, expected_md5):
                        return None
                else:
                    logger.info(f"{file_desc} exists with correct MD5")
            else:
                logger.info(f"{file_desc} exists (MD5 not verified)")
            return dest_path
        else:
            logger.info(f"{file_desc} not found, downloading...")
            if download_file(url, dest_path, expected_md5):
                return dest_path
            else:
                return None