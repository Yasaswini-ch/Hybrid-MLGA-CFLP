import numpy as np

class CFLPChromosome:
    """
    Encapsulates the discrete genetic representation (genotype) of a CFLP solution.
    
    A chromosome is a binary vector y of length m (num_facilities), where:
        y[i] = 1 indicates that facility i is open.
        y[i] = 0 indicates that facility i is closed.
    """
    
    def __init__(self, genes: np.ndarray):
        """
        Initializes the chromosome with a binary facility configuration.
        
        Args:
            genes (np.ndarray): NumPy binary integer array of shape (m,).
        """
        self.genes = np.array(genes, dtype=np.int32)
        self.size = len(self.genes)
        self.validate()
        
    def validate(self) -> None:
        """
        Validates the structural integrity of the chromosome.
        Ensures the vector contains strictly binary values (0 or 1).
        """
        if not np.all((self.genes == 0) | (self.genes == 1)):
            raise ValueError(f"Chromosome genes must be strictly binary (0 or 1). Got: {self.genes}")
            
    def active_count(self) -> int:
        """
        Returns the number of active (open) warehouses.
        
        Returns:
            int: The sum of the facility opening statuses.
        """
        return int(np.sum(self.genes))
        
    def hamming_distance(self, other: 'CFLPChromosome') -> int:
        """
        Calculates the bitwise Hamming distance between this chromosome and another.
        This represents the number of facilities in different opening states,
        serving as our primary metric for tracking population diversity.
        
        Args:
            other (CFLPChromosome): The other chromosome to compare.
            
        Returns:
            int: The number of bit differences (Hamming distance).
        """
        if self.size != other.size:
            raise ValueError(f"Cannot compute Hamming distance between chromosomes of different sizes: {self.size} vs {other.size}")
        return int(np.sum(self.genes != other.genes))
        
    def __str__(self) -> str:
        return f"CFLPChromosome(open={self.active_count()}/{self.size}, genes={self.genes.tolist()})"
        
    def __repr__(self) -> str:
        return self.__str__()
