import data from "../data/cap71.json";
export const D = data;

// Fractional greedy assignment for a given open-facility mask.
// Each customer fills its demand from the cheapest open facilities with spare capacity.
// (A fast stand-in for the LP transportation sub-problem the Python solver uses.)
export function evaluate(mask) {
  const { m, n, cap, fixed, dem, cost } = D;
  const rem = cap.slice();
  let fixedCost = 0, transport = 0, unmet = 0;
  for (let i = 0; i < m; i++) if (mask[i]) fixedCost += fixed[i];
  const flows = [];
  const order = [...Array(n).keys()].sort((a, b) => dem[b] - dem[a]);
  for (const j of order) {
    let need = dem[j];
    const opts = [];
    for (let i = 0; i < m; i++) if (mask[i]) opts.push(i);
    opts.sort((a, b) => cost[j][a] - cost[j][b]);
    for (const i of opts) {
      if (need <= 1e-9) break;
      const q = Math.min(need, rem[i]);
      if (q > 0) { rem[i] -= q; need -= q; transport += (q / dem[j]) * cost[j][i]; flows.push([j, i, q / dem[j]]); }
    }
    unmet += Math.max(0, need);
  }
  const penalty = unmet * 1000;
  return { total: fixedCost + transport + penalty, fixedCost, transport, unmet, feasible: unmet < 1e-6, flows, used: cap.map((c, i) => c - rem[i]) };
}

export function greedyOrder() {
  const { m, cap, fixed } = D;
  return [...Array(m).keys()].sort((a, b) => fixed[a] / cap[a] - fixed[b] / cap[b]);
}

// tiny seeded RNG so the demo is repeatable
export function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

export function runGA({ pop = 30, gens = 40, pc = 0.8, seed = 7 } = {}) {
  const { m } = D, r = rng(seed);
  const fit = (c) => evaluate(c).total;
  let P = Array.from({ length: pop }, () => Array.from({ length: m }, () => (r() < 0.6 ? 1 : 0)));
  let F = P.map(fit);
  const history = [];
  const snap = () => {
    const bi = F.indexOf(Math.min(...F));
    history.push({ best: F[bi], mean: F.reduce((a, b) => a + b, 0) / F.length, bestChrom: P[bi].slice(), pop: P.map((c) => c.slice()), fits: F.slice() });
  };
  snap();
  for (let g = 0; g < gens; g++) {
    const tour = () => { let b = -1; for (let k = 0; k < 3; k++) { const x = (r() * pop) | 0; if (b < 0 || F[x] < F[b]) b = x; } return P[b]; };
    const next = [];
    const elite = F.indexOf(Math.min(...F));
    next.push(P[elite].slice());
    while (next.length < pop) {
      let a = tour().slice(), b = tour().slice();
      if (r() < pc) { let x = (r() * m) | 0, y = (r() * m) | 0; if (x > y) [x, y] = [y, x]; for (let i = x; i <= y; i++) [a[i], b[i]] = [b[i], a[i]]; }
      for (const c of [a, b]) { for (let i = 0; i < m; i++) if (r() < 1 / m) c[i] ^= 1; if (next.length < pop) next.push(c); }
    }
    P = next; F = P.map(fit); snap();
  }
  return history;
}
