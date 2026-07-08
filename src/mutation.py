import random
from typing import Tuple

def bit_flip_mutation(individual: list, indpb: float) -> Tuple[list,]:
    """
    Executes a Bit-Flip Mutation in-place on a binary chromosome.
    
    Algorithm:
        For each gene (bit) in the chromosome, flip its value (0 -> 1 or 1 -> 0)
        with an independent probability of 'indpb'.
        
    Args:
        individual (list): The individual chromosome to mutate.
        indpb (float): Independent probability for each gene to be flipped.
        
    Returns:
        Tuple[list,]: A 1-tuple containing the modified individual.
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = 1 - individual[i]
            
    return (individual,)
