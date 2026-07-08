# Mixed-Integer Linear Programming Formulation: Capacitated Facility Location Problem (CFLP)

This document provides a formal, researcher-grade Mixed-Integer Linear Programming (MILP) formulation of the Capacitated Facility Location Problem (CFLP) used in our logistics network design and evolutionary computation research.

---

## 1. Physical Analogy & Beginner-Friendly Concept

Imagine you are the Operations and Logistics Director for an industrial supply chain:

1.  **Facility Opening Decisions**: You have a set of potential warehouse locations ($m$). Opening a warehouse is expensive; it incurs a fixed overhead cost ($f_i$) to rent the property, hire staff, and maintain utility networks. You must decide which warehouses to open ($y_i = 1$) and which to leave closed ($y_i = 0$).
2.  **Customer Assignment & Transportation Costs**: You have a set of retail stores ($n$), each with a specific product demand ($d_j$). Shipping products from warehouse $i$ to store $j$ costs money; the shipping cost scales with distance and volume ($c_{ij} \times x_{ij}$, where $x_{ij}$ is the quantity shipped).
3.  **Capacity Constraints**: Each warehouse has a physical limit on the volume it can store ($s_i$). You cannot ship more product out of a warehouse than its capacity. If a warehouse is closed ($y_i = 0$), you cannot ship anything from it.
4.  **Goal**: Minimize your total budget, which is the sum of **fixed opening costs** and **variable transportation costs**, while satisfying 100% of all store demands.

---

## 2. Formal Mathematical Formulation

### Sets and Indices
- $I$: Set of potential facility locations, indexed by $i \in I = \{1, 2, \dots, m\}$
- $J$: Set of customers, indexed by $j \in J = \{1, 2, \dots, n\}$

### Parameters
- $f_i$: Fixed cost incurred to open facility $i$ (currency units)
- $s_i$: Capacity of facility $i$ (the maximum supply volume it can provide)
- $d_j$: Demand of customer $j$ (product units)
- $c_{ij}$: Unit transportation cost from facility $i$ to customer $j$ (cost per unit shipped)

### Decision Variables
- $y_i \in \{0, 1\}$: Binary variable indicating if facility $i$ is open ($y_i = 1$) or closed ($y_i = 0$).
- $x_{ij} \ge 0$: Continuous flow variable representing the quantity of demand for customer $j$ supplied by facility $i$.

---

### Objective Function

The objective is to minimize the sum of fixed opening costs and total variable transportation costs:

$$\min \quad Z = \sum_{i \in I} f_i y_i + \sum_{j \in J} \sum_{i \in I} c_{ij} x_{ij}$$

*   **Fixed Costs Component**: $\sum_{i \in I} f_i y_i$  
    The overhead cost incurred by opening facilities. Closed facilities ($y_i = 0$) contribute $\$0$ to this term.
*   **Variable Transportation Costs Component**: $\sum_{j \in J} \sum_{i \in I} c_{ij} x_{ij}$  
    The total shipping cost incurred by routing product flows.

---

### Constraints

#### 1. Demand Satisfaction
Every customer's demand must be fully met by the open facilities:
$$\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J$$
*   *Operations Logic*: Customers cannot receive deficit shipments. The total quantity shipped from all potential warehouses to customer $j$ must exactly equal customer $j$'s demand ($d_j$).

#### 2. Capacity Limits
The total flow supplied by a facility to all customers cannot exceed its capacity. If the facility is closed ($y_i = 0$), no flow can originate from it:
$$\sum_{j \in J} x_{ij} \le s_i y_i \quad \forall i \in I$$
*   *Operations Logic*:
    - If facility $i$ is closed ($y_i = 0$), the right-hand side becomes $0$. The sum of flows $\sum_{j \in J} x_{ij}$ is forced to $\le 0$, which (coupled with non-negativity) implies no flows can originate from it.
    - If facility $i$ is open ($y_i = 1$), the right-hand side is $s_i$. The total quantity shipped from warehouse $i$ to all customers cannot exceed its physical storage limit.

#### 3. Domain Restrictions
Flows must be non-negative, and facility status must be binary:
$$x_{ij} \ge 0 \quad \forall i \in I, \forall j \in J$$
$$y_i \in \{0, 1\} \quad \forall i \in I$$
*   *Operations Logic*: You cannot ship negative quantities, and a warehouse cannot be "semi-open".

---

## 3. Feasible vs. Infeasible Landscapes

Understanding the boundaries of feasibility is highly vital when designing optimization algorithms (especially Genetic Algorithms, where random mutations can easily create invalid configurations):

| Parameter | Feasible Solution | Infeasible Solution |
| :--- | :--- | :--- |
| **Demand Met** | Total flow to customer $j$ equals $d_j$ for all $j \in J$. | A customer's demand is under-satisfied or over-satisfied. |
| **Capacity Check** | Total shipping volume from warehouse $i$ is $\le s_i$ for all open warehouses. | A warehouse ships more product than its physical storage capacity. |
| **Facility State** | Product flows originate only from open warehouses ($y_i = 1$). | Product flows originate from a closed warehouse ($y_i = 0$). |
| **Feasibility Region** | Serviced by at least the minimum number of open warehouses required to cover total network demand: $K \ge \lceil \sum d_j / s_i \rceil$. | The number of active warehouses is too small to physically cover the sum of customer demands. |

---

## 4. Connection to Genetic Algorithms & Surrogate Modeling

This formal formulation establishes the foundation for our upcoming evolutionary and machine learning optimization stages:

1.  **Chromosome Design**: We will represent a candidate solution as a simple binary chromosome vector `y` of length $m$ representing the facility status.
2.  **Fitness Evaluation Bottleneck**: For a given facility chromosome `y`, we must solve a continuous linear programming flow allocation sub-problem:
    $$\min_{x} \sum_{j \in J} \sum_{i \in I} c_{ij} x_{ij} \quad \text{subject to} \quad \sum_{i \in I} x_{ij} = d_j, \quad \sum_{j \in J} x_{ij} \le s_i y_i, \quad x_{ij} \ge 0$$
    Solving this allocation sub-problem for every chromosome in every generation is the computational bottleneck of GAs.
3.  **Surrogate Predictor**: Our Random Forest regressor will learn to predict the total cost of a chromosome directly from the binary status `y`, bypassing the expensive continuous assignment step and speeding up evolutionary loops by several orders of magnitude.
