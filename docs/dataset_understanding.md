# Research Note: Dataset Understanding & Mathematical Formulation

## 1. Mathematical Formulation of the Capacitated Facility Location Problem (CFLP)

The Capacitated Facility Location Problem (CFLP) is a classic NP-hard combinatorial optimization problem in logistics, operations research, and supply chain management. It can be formulated as a Mixed-Integer Linear Program (MILP).

### Sets and Indices
- $I$: Set of potential facility locations, indexed by $i \in I = \{1, 2, \dots, m\}$
- $J$: Set of customers, indexed by $j \in J = \{1, 2, \dots, n\}$

### Parameters
- $f_i$: Fixed cost incurred to open facility $i$
- $s_i$: Capacity of facility $i$ (the maximum supply it can provide)
- $d_j$: Demand of customer $j$
- $c_{ij}$: Unit transportation cost from facility $i$ to customer $j$

### Decision Variables
- $y_i \in \{0, 1\}$: Binary variable indicating if facility $i$ is open ($y_i = 1$) or closed ($y_i = 0$).
- $x_{ij} \ge 0$: Continuous flow variable representing the quantity of demand for customer $j$ supplied by facility $i$.

---

### Objective Function
Minimize the sum of fixed opening costs and total variable transportation costs:

$$\min \quad Z = \sum_{i \in I} f_i y_i + \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$$

---

### Constraints

#### 1. Demand Satisfaction
Every customer's demand must be fully met by the open facilities:
$$\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$$

#### 2. Capacity Limits
The total flow supplied by a facility to all customers cannot exceed its capacity. If the facility is closed ($y_i = 0$), no flow can originate from it:
$$\sum_{j \in J} x_{ij} \le s_i y_i \quad \forall i \in I$$

#### 3. Domain Restrictions
Flows must be non-negative, and facility status must be binary:
$$x_{ij} \ge 0 \quad \forall i \in I, \forall j \in J$$
$$y_i \in \{0, 1\} \quad \forall i \in I$$

---

## 2. Beasley OR-Library Dataset Structure

The benchmark datasets used in this project are retrieved from the **OR-Library** (J.E. Beasley, 1990). Specifically, we are using the `cap41` to `cap44` files. 

### General Structural Layout

An OR-Library CFLP text file is structured sequentially as follows:

| Line Group | Content | Description | Data Type |
| :--- | :--- | :--- | :--- |
| **Line 1** | `$m \quad n$` | Number of facilities $m$ ($I$) and number of customers $n$ ($J$). | Space-separated integers |
| **Lines 2 to $m+1$** | `$s_i \quad f_i$` | Capacity $s_i$ and fixed opening cost $f_i$ for facility $i$. | Space-separated integer and float |
| **Next $n$ Blocks** | Customer Demands & Costs | Repeated $n$ times (once per customer $j = 1 \dots n$): | Mixed |
| *Block Header* | `$d_j$` | Demand of customer $j$. | Single integer |
| *Block Data* | `$c_{1j} \dots c_{mj}$` | Unit transportation costs from each facility $1 \dots m$ to customer $j$. Distributed across multiple lines in groups of up to 7 space-separated floats. | Space-separated floats |

---

### In-Depth Example: Analysis of `cap41.txt`

Let's dissect `cap41.txt`:
- **Number of Facilities ($m$):** 16
- **Number of Customers ($n$):** 50
- **Facilities capacity and costs (16 lines):**
  - Most facilities have a capacity $s_i = 5000$ and fixed cost $f_i = 7500.0$.
  - Facility index 10 (line 12: `5000 0.`) has an opening cost of $0.0$. This represents a "mandatory" or already active facility that is free to open.
- **Customer Demand and Transportation Costs:**
  - Customer 1:
    - Demand ($d_1$): `146`
    - Costs ($c_{i,1}$): 16 values spread across 3 lines:
      - Line 19: `6739.72500 10355.05000 7650.40000 5219.50000 5776.12500 6641.17500 4374.52500` (7 values)
      - Line 20: `3847.10000 6429.47500 5396.52500 5219.50000 4182.90000 7391.25000 5038.82500` (7 values)
      - Line 21: `10349.57500 6051.70000` (2 values)
      - Total = 16 values, representing unit shipping costs from facility 1 to 16.
  - Customer 2:
    - Demand ($d_2$): `87`
    - Costs ($c_{i,2}$): 16 values on lines 23-25.

### Benchmark File Grid Comparison

The benchmark instances differ in fixed facility opening costs and capacities, which tests the sensitivity of optimization algorithms to fixed-versus-variable cost ratios and constraint tightness.

| File | Facilities ($m$) | Customers ($n$) | Facility Capacity ($s_i$) | Standard Fixed Cost ($f_i$) | Special Facility Cost ($f_{11}$) | Capacity/Demand Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **cap41.txt** | 16 | 50 | 5,000 | $7,500.0 | $0.0 | 1.3730 |
| **cap42.txt** | 16 | 50 | 5,000 | $12,500.0 | $0.0 | 1.3730 |
| **cap43.txt** | 16 | 50 | 5,000 | $17,500.0 | $0.0 | 1.3730 |
| **cap44.txt** | 16 | 50 | 5,000 | $25,000.0 | $0.0 | 1.3730 |
| **cap51.txt** | 16 | 50 | 10,000 | $17,500.0 | $0.0 | 2.7459 |
| **cap61.txt** | 16 | 50 | 15,000 | $7,500.0 | $0.0 | 4.1189 |
| **cap62.txt** | 16 | 50 | 15,000 | $12,500.0 | $0.0 | 4.1189 |
| **cap63.txt** | 16 | 50 | 15,000 | $17,500.0 | $0.0 | 4.1189 |
| **cap64.txt** | 16 | 50 | 15,000 | $25,000.0 | $0.0 | 4.1189 |
| **cap71.txt** | 16 | 50 | 58,268 | $7,500.0 | $0.0 | 16.0000 |
| **cap72.txt** | 16 | 50 | 58,268 | $12,500.0 | $0.0 | 16.0000 |
| **cap73.txt** | 16 | 50 | 58,268 | $17,500.0 | $0.0 | 16.0000 |
| **cap74.txt** | 16 | 50 | 58,268 | $25,000.0 | $0.0 | 16.0000 |
| **cap81.txt** | 25 | 50 | 5,000 | $7,500.0 | $0.0 | 2.1453 |
| **cap82.txt** | 25 | 50 | 5,000 | $12,500.0 | $0.0 | 2.1453 |
| **cap83.txt** | 25 | 50 | 5,000 | $17,500.0 | $0.0 | 2.1453 |
| **cap84.txt** | 25 | 50 | 5,000 | $25,000.0 | $0.0 | 2.1453 |
| **cap91.txt** | 25 | 50 | 15,000 | $7,500.0 | $0.0 | 6.4358 |
| **cap92.txt** | 25 | 50 | 15,000 | $12,500.0 | $0.0 | 6.4358 |
| **cap93.txt** | 25 | 50 | 15,000 | $17,500.0 | $0.0 | 6.4358 |
| **cap94.txt** | 25 | 50 | 15,000 | $25,000.0 | $0.0 | 6.4358 |
| **cap101.txt** | 25 | 50 | 58,268 | $7,500.0 | $0.0 | 25.0000 |
| **cap102.txt** | 25 | 50 | 58,268 | $12,500.0 | $0.0 | 25.0000 |
| **cap103.txt** | 25 | 50 | 58,268 | $17,500.0 | $0.0 | 25.0000 |
| **cap104.txt** | 25 | 50 | 58,268 | $25,000.0 | $0.0 | 25.0000 |
| **cap111.txt** | 50 | 50 | 5,000 | $7,500.0 | $7,500.0 | 4.2905 |
| **cap112.txt** | 50 | 50 | 5,000 | $12,500.0 | $12,500.0 | 4.2905 |
| **cap113.txt** | 50 | 50 | 5,000 | $17,500.0 | $17,500.0 | 4.2905 |
| **cap114.txt** | 50 | 50 | 5,000 | $25,000.0 | $25,000.0 | 4.2905 |
| **cap121.txt** | 50 | 50 | 15,000 | $7,500.0 | $7,500.0 | 12.8716 |
| **cap122.txt** | 50 | 50 | 15,000 | $12,500.0 | $12,500.0 | 12.8716 |
| **cap123.txt** | 50 | 50 | 15,000 | $17,500.0 | $17,500.0 | 12.8716 |
| **cap124.txt** | 50 | 50 | 15,000 | $25,000.0 | $25,000.0 | 12.8716 |
| **cap131.txt** | 50 | 50 | 58,268 | $7,500.0 | $7,500.0 | 50.0000 |
| **cap132.txt** | 50 | 50 | 58,268 | $12,500.0 | $12,500.0 | 50.0000 |
| **cap133.txt** | 50 | 50 | 58,268 | $17,500.0 | $17,500.0 | 50.0000 |
| **cap134.txt** | 50 | 50 | 58,268 | $25,000.0 | $25,000.0 | 50.0000 |

---

## 3. Generalization & Cross-Problem Set Analysis (PS IV vs. PS V vs. PS VI vs. PS VII vs. PS VIII vs. PS IX vs. PS X vs. PS XI vs. PS XII vs. PS XIII)

An essential milestone in developing metaheuristics is **benchmark generalization**—ensuring the algorithm and parsers execute robustly across different problem sets without parameter hardcoding.

### Structural Comparison: Problem Sets IV through XIII

1. **Topological and Dimensional Scaling**:
   - **Problem Sets IV, V, VI, and VII**: Share a fixed $16 \times 50$ dimension ($m=16$ facilities, $n=50$ customers).
   - **Problem Sets VIII (`cap81-84`), IX (`cap91-94`), and X (`cap101-104`)**: Scale candidate facility locations to **$m=25$ facilities** (a **56.25% increase** in candidates), while keeping customers fixed at $n=50$.
   - **Problem Sets XI (`cap111-114`), XII (`cap121-124`), and XIII (`cap131-134`)**: Scale candidate facility locations to **$m=50$ facilities** (a massive **100% increase** compared to the 25-facility space, and **212.5% increase** compared to the 16-facility space), while keeping customers fixed at $n=50$. This represents the absolute highest dimensional complexity in this benchmark pipeline, testing the true dynamic robustness of the parser and solver logic under extreme scaling.
   - The customer demands ($d_j$) are **100% identical** across all 37 instances, and the transportation cost sub-matrices are identical for the first 16 facilities.

2. **Capacity Scaling & Constraint Tightness**:
   - **Problem Set IV (`cap41-44`)**: Capacity $s_i = 5,000$. Tightness ratio is **1.3730**. At least **12 facilities** must be open ($\lceil 58,268 / 5,000 \rceil = 12$).
   - **Problem Set V (`cap51`)**: Capacity $s_i = 10,000$. Tightness ratio is **2.7459**. At least **6 facilities** must be open.
   - **Problem Set VI (`cap61-64`)**: Capacity $s_i = 15,000$. Tightness ratio is **4.1189**. At least **4 facilities** must be open.
   - **Problem Set VII (`cap71-74`)**: Capacity $s_i = 58,268$. Tightness ratio is **16.0000** (uncapacitated boundary). At least **1 facility** must be open.
   - **Problem Set VIII (`cap81-84`)**: Capacity $s_i = 5,000$. Tightness ratio is **2.1453**. At least **12 facilities** must be open ($\lceil 58,268 / 5,000 \rceil = 12$).
   - **Problem Set IX (`cap91-94`)**: Capacity $s_i = 15,000$. Tightness ratio is **6.4358**. At least **4 facilities** must be open ($\lceil 58,268 / 15,000 \rceil = 4$).
   - **Problem Set X (`cap101-104`)**: Capacity $s_i = 58,268$ (uncapacitated UFLP limit). Tightness ratio is **25.0000**. At least **1 facility** must be open ($\lceil 58,268 / 58,268 \rceil = 1$).
   - **Problem Set XI (`cap111-114`)**: Capacity $s_i = 5,000$. Tightness ratio is **4.2905**. At least **12 facilities** must be open to satisfy demand ($\lceil 58,268 / 5,000 \rceil = 12$).
   - **Problem Set XII (`cap121-124`)**: Capacity $s_i = 15,000$. Tightness ratio is **12.8716** (loose capacity limits). At least **4 facilities** must be open to satisfy demand ($\lceil 58,268 / 15,000 \rceil = 4$).
   - **Problem Set XIII (`cap131-134`)**: Capacity $s_i = 58,268$ (uncapacitated UFLP limit). Tightness ratio is **50.0000**. At least **1 facility** must be open ($\lceil 58,268 / 58,268 \rceil = 1$).

3. **Fixed Opening Cost Scaling ($f_i$)**:
   - Problem Sets VIII through XIII mirror the four-tier standard cost structures of prior sets, testing opening costs at cheap ($\$7,500$ in `cap81`/`cap91`/`cap101`/`cap111`/`cap121`/`cap131`), moderate ($\$12,500$ in `cap82`/`cap92`/`cap102`/`cap112`/`cap122`/`cap132`), high ($\$17,500$ in `cap83`/`cap93`/`cap103`/`cap113`/`cap123`/`cap133`), and expensive ($\$25,000$ in `cap84`/`cap94`/`cap104`/`cap114`/`cap124`/`cap134`) levels.
   - **Free Facility Shift**: Unlike prior sets (Problem Sets IV through X) where the free facility ($f_i = \$0.0$) was located at index 10, Problem Set XIII shifts the free facility to **index 22 (facility 23)**. Problem Sets XI and XII contain no free facilities.

### Research Implications on Algorithm Complexity

- **Feasible Combinatorial Landscape**: 
  In a binary facility opening space of $2^{50} \approx 1.125 \times 10^{15}$ configurations (over **1.12 quadrillion combinations**!):
  - Scaling potential facility locations to 50 expands the binary search space by **33,554,432-fold** compared to $m=25$ spaces. This represents the ultimate exploratory challenge for our Genetic Algorithm, making baseline benchmarks and dynamic parser validation highly relevant.
  - In `cap131-134`, only 1 open facility is required, so **every single non-zero binary vector is 100% physically feasible**. This simplifies the search space, transforming UFLP into a pure unconstrained binary optimization landscape. This represents a critical testing ground to evaluate the crossover and mutation operators of our Genetic Algorithm.
- **The Trade-Off of Marginal Facility Utility**:
  - The exact MILP solver found that rather than opening all 50 facilities, it is optimal to leave a few closed. It opens **47 / 50** in `cap131` and drops to **45 / 50** in `cap134` as fixed costs rise.
  - *OR Equivalence Discovery:* The optimal active facility configurations and costs for Problem Set XIII are **identical to the penny** to those in Problem Set XII (e.g. `cap131` and `cap121` both cost \$2,850,307,905.40). Since MILP opens 45 to 47 facilities under both sets, capacity constraints are already completely non-binding. The increase in capacity from 15,000 (PS XII) to 58,268 (PS XIII) does not alter the optimal routing paths or fixed opening costs.
  - However, the Greedy heuristic suffered a catastrophic collapse under PS XIII, yielding a staggering **99.98% optimality gap** ($>\$2.849$ billion in wasted cost!). Greedy is blinded by standard fixed costs, opening exactly **1 single facility** (facility 23 at index 22) to zero out standard opening costs, while completely ignoring the enormous routing penalties of not using the remaining 24 nearby facilities.




