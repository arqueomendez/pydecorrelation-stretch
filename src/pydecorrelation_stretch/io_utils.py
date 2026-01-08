
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

logger = logging.getLogger(__name__)

class ImageLoadingResult:
    def __init__(self, data, is_memmap: bool, temp_file_path: str = None):
        self.data = data
        self.is_memmap = is_memmap
        self.temp_file_path = temp_file_path
        
    def cleanup(self):
        """Cleanup temporary files if memmap."""
        if self.is_memmap and self.temp_file_path:
            # We must delete the object first to close file handle generally,
            # but in numpy memmap, closing is tricky.
            # Best effort cleanup.
            try:
                # Force delete reference to close handle?
                # On Windows, we can't delete file if open.
                # Numpy memmap doesn't have a close() method.
                if hasattr(self.data, '_mmap'):
                     self.data._mmap.close()
                del self.data
            except:
                pass
                
            if os.path.exists(self.temp_file_path):
                try:
                    os.remove(self.temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temp memmap file: {e}")

def get_available_memory_mb():
    return psutil.virtual_memory().available / (1024 * 1024)

def smart_load_image(file_path: str) -> ImageLoadingResult:
    """
    Load image, automatically choosing between RAM and Memory Map (Disk)
    based on available system memory.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")
        
    # Check dimensions lazily with PIL
    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            
            if mode != 'RGB':
                # Estimate size for RGB conversion
                channels = 3
            else:
                channels = 3 # RGB
                
            # Estimated uncompressed size in MB
            pixels = width * height
            estimated_size_mb = (pixels * channels) / (1024 * 1024)
            
            available_ram_mb = get_available_memory_mb()
            
            # Safety Threshold: Use max 50% of AVAILABLE RAM
            threshold_mb = available_ram_mb * 0.5
            
            logger.info(f"Image Size: {width}x{height} (~{estimated_size_mb:.2f} MB)")
            logger.info(f"Available RAM: {available_ram_mb:.2f} MB (Threshold: {threshold_mb:.2f} MB)")
            
            if estimated_size_mb > threshold_mb:
                logger.warning("Image too large for RAM. switching to Disk-backed Memory Map (Streaming Load)...")
                return _load_to_memmap(path, width, height)
            else:
                logger.info("Loading to RAM...")
                # Re-open to load content
                with Image.open(path) as full_img:
                    if full_img.mode != 'RGB':
                        full_img = full_img.convert('RGB')
                    return ImageLoadingResult(np.array(full_img, dtype=np.uint8), False)
                    
    except Exception as e:
        logger.error(f"Error checking image size: {e}")
        # Fallback to standard load
        img = Image.open(path).convert('RGB')
        return ImageLoadingResult(np.array(img, dtype=np.uint8), False)

def _load_to_memmap(file_path: Path, width: int, height: int) -> ImageLoadingResult:
    """
    Load huge image into a memory mapped file by processing chunks.
    """
    # Create temp file
    fd, temp_path = tempfile.mkstemp()
    os.close(fd)
    
    shape = (height, width, 3)
    
    # Create memmap
    fp = np.memmap(temp_path, dtype='uint8', mode='w+', shape=shape)
    
    # Constants
    CHUNK_HEIGHT = 1024 # Rows per chunk
    
    try:
        # Re-open PIL image
        with Image.open(file_path) as img:
            rgb_img = img if img.mode == 'RGB' else img.convert('RGB')
            
            start_row = 0
            while start_row < height:
                end_row = min(start_row + CHUNK_HEIGHT, height)
                rows_to_read = end_row - start_row
                
                # Crop logic: (left, upper, right, lower)
                # Note: crop is lazy? crop().load() forces load.
                # Optimized way:
                box = (0, start_row, width, end_row)
                chunk_img = rgb_img.crop(box)
                
                # Convert to numpy and write to memmap
                chunk_arr = np.array(chunk_img, dtype=np.uint8)
                
                # Assign to memmap
                fp[start_row:end_row, :] = chunk_arr
                
                # Flush every few chunks?
                if start_row % (CHUNK_HEIGHT * 10) == 0:
                     fp.flush()
                     
                start_row += rows_to_read
                print(f"Loaded rows {start_row}/{height}...", end='\r')
                
        fp.flush()
        print("\nMemmap load complete.")
        
        # Re-open as read-only (or read-copy-on-write?)
        # Keep it r+ or w+ to allow modifications if needed?
        # Actually, dstretch usually returns NEW image.
        # So input can be 'r'.
        # But we return the 'fp' object which is already open.
        
        return ImageLoadingResult(fp, True, temp_path)
        
    except Exception as e:
        # Cleanup if failed
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        raise e
