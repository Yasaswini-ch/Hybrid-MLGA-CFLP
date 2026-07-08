# Genetic Algorithm Design Notes: Modular Parameters, Mechanics, & Repair Theory

This document details the engineering specifications, mathematical formulas, and algorithmic designs for our modular classical Genetic Algorithm (GA) implementation for the Capacitated Facility Location Problem (CFLP).

---

## 1. Algorithmic Specifications

### Chromosome Wrapper (`chromosome.py`)
Rather than dealing with raw Python lists, we wrap the binary facility vector in a specialized class `CFLPChromosome`:
*   **Properties**:
    *   `genes`: NumPy binary array of shape $(m,)$ where $y_i \in \{0, 1\}$.
    *   `size`: Length of the chromosome ($m$).
*   **Methods**:
    *   `validate()`: Ensures size matches the dataset and values are strictly binary.
    *   `hamming_distance(other)`: Computes the bitwise differences between two chromosomes to track population diversity.
    *   `active_count()`: Returns the number of open warehouses ($\sum y_i$).

### Population Initialization (`population.py`)
To study convergence patterns, we implement two distinct population initialization strategies:
1.  **Purely Random Binary Initialization**:
    *   Each facility bit is initialized randomly to $0$ or $1$ with a probability $p=0.5$.
    *   *Scientific Value*: Represents a standard unbiased starting state, testing the GA's ability to navigate from highly chaotic, infeasible starting structures.
2.  **Heuristic-Sorted Seeding (Smart Initialization)**:
    *   Computes the physical lower limit of active warehouses required for feasibility:
        $$m_{min} = \left\lceil \frac{\sum d_j}{\max(s_i)} \right\rceil$$
    *   Ensures that every generated individual has at least $m_{min}$ facilities set to $1$.
    *   *Scientific Value*: Introduces localized domain knowledge to seed the starting population in highly promising, feasible regions of the landscape, accelerating convergence rates.

---

## 2. Selection, Crossover, & Mutation Configurations

To support rigorous experimental research, we decouple our operators into modular functions under `src/`:

### Selection Operators (`selection.py`)
We implement two core selection operators to compare selection pressure:
1.  **Tournament Selection**:
    *   *Parameter*: `tournsize` ($k$).
    *   *Logic*: Select $k$ individuals at random and choose the best one.
    *   *Research Insight*: A higher tournament size (e.g., $k=5$) leads to aggressive exploitation, which can cause premature convergence. We set $k=3$ as our robust research default.
2.  **Roulette Wheel Selection**:
    *   *Logic*: Select individuals with a probability proportional to their fitness. Since CFLP is a minimization problem, we convert fitness values using an offset:
        $$f'_{k} = Z_{max} - Z_k + \epsilon$$
        $$P(k) = \frac{f'_k}{\sum f'_i}$$
    *   *Research Insight*: Demonstrates how weak selection pressure can lead to slow convergence but highly diverse populations.

### Crossover Operators (`crossover.py`)
We implement three crossover configurations to evaluate genetic recombination:
1.  **Single-Point Crossover**: Recombines parents at a single index.
2.  **Two-Point Crossover**: Recombines parents using a middle segment. This is our default due to its lower disruptive property.
3.  **Uniform Crossover**: Each gene is inherited from Parent 1 or Parent 2 with equal probability.

### Mutation Operators (`mutation.py`)
1.  **Bit-Flip Mutation**:
    *   *Parameters*: `mut_pb` (probability of mutating an individual) and `indpb` (probability of mutating a single bit within that individual, set to $1/m$).
    *   *Logic*: Each bit has a probability `indpb` of being inverted.

---

## 3. Lamarckian Feasibility Repair Theory (`repair.py`)

Constraint handling is the most critical component of applying metaheuristics to the CFLP. Since random crossovers and mutations frequently break physical capacity constraints, we implement a **Lamarckian Feasibility Repair Operator**.

### Mathematical Formulation of Lamarckian Repair
Let $\mathbf{y} \in \{0, 1\}^m$ be a binary facility status vector. Let $s_i$ be the capacity of facility $i$, and $d_j$ be the demand of customer $j$.
An individual is physically infeasible if:
$$\sum_{i=1}^m s_i y_i < \sum_{j=1}^n d_j$$

Our repair operator works as follows:
1.  Compute the total customer demand: $D = \sum_{j=1}^n d_j$.
2.  Compute the currently active capacity: $C(\mathbf{y}) = \sum_{i=1}^m s_i y_i$.
3.  If $C(\mathbf{y}) < D$:
    *   Define the set of closed facilities: $I_{closed} = \{i \mid y_i = 0\}$.
    *   For each facility $i \in I_{closed}$, compute the **efficiency ratio**:
        $$E_i = \frac{f_i}{s_i}$$
        where $f_i$ is the fixed cost and $s_i$ is the capacity.
    *   Sort the facilities in $I_{closed}$ in ascending order of efficiency:
        $$\text{Sorted } I_{closed} = [i_{(1)}, i_{(2)}, \dots, i_{(k)}]$$
        such that $E_{i_{(1)}} \le E_{i_{(2)}} \le \dots \le E_{i_{(k)}}$.
    *   Sequentially open facilities from the sorted list:
        $$y_{i_{(p)}} \leftarrow 1$$
        Update active capacity: $C(\mathbf{y}) \leftarrow C(\mathbf{y}) + s_{i_{(p)}}$.
        Repeat this step until $C(\mathbf{y}) \ge D$.
4.  **Lamarckian Genetic Propagation**: Modify the original individual's chromosome in-place with the repaired facility vector $\mathbf{y}$.

### Why Lamarckian Repair is Superior to Pure Penalty
In highly constrained problem sets (like PS IV `cap41` with tight capacity ratios), the ratio of feasible configurations is extremely small. 
*   Under a **Pure Penalty approach**, if a mutation produces an infeasible individual, its fitness is set to a massive penalty (e.g., $10^{12}$). The individual is immediately eliminated by selection. This acts as a "death penalty," destroying valuable genetic sub-structures (schema) that may have been highly optimal in other aspects.
*   Under **Lamarckian Repair**, we actively guide the search back to the boundary of the feasible region. By using an efficiency-sorted heuristic, we ensure that the added facilities are highly cost-effective, creating a powerful synergy between global evolutionary exploration and local heuristic refinement.
