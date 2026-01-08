

import numpy as np


class StreamingCovariance:
    """
    Computes mean and covariance matrix incrementally (streaming).
    Used for processing images larger than available RAM.
    """
    
    def __init__(self, n_features: int = 3):
        self.n_features = n_features
        self.count = 0
        # Welford's algorithm or simple Sum accumulation?
        # For Covariance Matrix (X.T @ X), Sums are sufficient and faster/simpler for vectorization.
        # Cov(X) = E[X.T @ X] - E[X].T @ E[X]
        # We need:
        # 1. Sum of X (for Mean)
        # 2. Sum of X.T @ X (for Second Moment)
        
        self.sum_x = np.zeros(n_features, dtype=np.float64)
        self.sum_xxT = np.zeros((n_features, n_features), dtype=np.float64)
        
    def update(self, chunk: np.ndarray):
        """
        Update statistics with a new chunk of data.
        
        Args:
            chunk: Array of shape (N, n_features)
        """
        # Ensure float64 for precision accumulation
        chunk_f64 = chunk.astype(np.float64)
        
        # 1. Update Count
        n = chunk_f64.shape[0]
        self.count += n
        
        # 2. Update Sum X
        self.sum_x += np.sum(chunk_f64, axis=0)
        
        # 3. Update Sum XXT
        # Optimization: chunk.T @ chunk is fast in BLAS
        self.sum_xxT += chunk_f64.T @ chunk_f64
        
    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate final Mean and Covariance Matrix.
        
        Returns:
            (mean, covariance)
        """
        if self.count <= 1:
            return self.sum_x * 0, self.sum_xxT * 0
            
        # Mean
        mean = self.sum_x / self.count
        
        # Covariance
        # Cov = (Sum_XXT - N * mean.T @ mean) / (N - 1)
        # Or more numerically stable: (Sum_XXT - (Sum_X * Sum_X.T) / N) / (N - 1)
        
        term2 = np.outer(self.sum_x, self.sum_x) / self.count
        covariance = (self.sum_xxT - term2) / (self.count - 1)
        
        return mean.astype(np.float32), covariance.astype(np.float32)

def calculate_streaming_statistics(
    image_source, # array or memmap
    chunk_size: int = 1_000_000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper to run streaming covariance on an image source.
    """
    # Reshape to list of pixels without copy (if possible)
    # Be careful with memmap, reshape might trigger read if not contiguous? 
    # Usually reshape on (H,W,C) to (-1, C) is O(1) stride manipulation.
    
    flat_source = image_source.reshape(-1, 3)
    total_pixels = flat_source.shape[0]
    
    streamer = StreamingCovariance(n_features=3)
    
    # Iterate in chunks
    for i in range(0, total_pixels, chunk_size):
        # Slicing a memmap returns a memmap (disk backed)
        # We copy to RAM only the chunk when calling 'update' (via astype inside)
        chunk = flat_source[i : i + chunk_size]
        streamer.update(chunk)
        
    return streamer.finalize()
