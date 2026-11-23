import React, { useMemo, useState, useEffect } from "react";
import { D, evaluate, greedyOrder, runGA } from "../lib/cflp.js";
import { money, pct, secs } from "../lib/fmt.js";
import { FacilityMap, LineChart, Chip, Chromosome } from "../components.jsx";
import results from "../data/results.json";

const allOpen = () => Array(D.m).fill(1);
const noneOpen = () => Array(D.m).fill(0);
const maskOf = (ids) => { const m = noneOpen(); ids.forEach((i) => (m[i] = 1)); return m; };
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
const cap71 = results.instances.find((x) => x.name === "cap71");
const sum = (a) => a.reduce((x, y) => x + y, 0);

/* ============ 0. Intro ============ */
function Card({ n, title, children, on }) {
  return (
    <div style={{ flex: 1, minWidth: 220, padding: 16, border: "1.5px solid var(--ink)", borderRadius: 12, boxShadow: "var(--pop)",
      background: on ? "var(--green)" : "var(--paper-2)", transition: ".25s", transform: on ? "translateY(-4px)" : "none" }}>
      <div style={{ fontSize: 11, color: "var(--muted)" }}>{n}</div>
      <b style={{ fontSize: 16 }}>{title}</b>
      <p style={{ margin: "6px 0 0" }}>{children}</p>
    </div>
  );
}
function Intro({ step }) {
  return (
    <div className="stage">
      {step === 0 && (
        <>
          <p style={{ fontSize: 16, maxWidth: "62ch" }}>
            A supply chain must decide which candidate facilities (warehouses or distribution centres) to open, and which open facility serves each customer.
            Every facility has a fixed opening cost and a limited capacity, and every shipment has a transportation cost.
          </p>
          <div className="row">
            <Card n="OPEN TOO MANY" title="High fixed cost">Opening cost is paid even for facilities that are barely used.</Card>
            <Card n="OPEN TOO FEW" title="High transport cost">Customers are served from farther away, and capacity limits can make the plan infeasible.</Card>
          </div>
          <div className="note">This is the Capacitated Facility Location Problem (CFLP). It is NP-hard: the number of possible facility configurations grows exponentially with the number of candidate sites.</div>
        </>
      )}
      {step === 1 && (
        <div className="row">
          <Card n="01" title="Problem" on>Choose the set of open facilities that minimizes total cost: fixed opening costs plus transportation cost, subject to capacity and demand constraints.</Card>
          <Card n="02" title="Method" on>A genetic algorithm searches over open/closed configurations using selection, crossover and mutation.</Card>
          <Card n="03" title="Contribution" on>A machine-learning surrogate model predicts the cost of a configuration, so the expensive exact evaluation is run only for promising candidates.</Card>
        </div>
      )}
      {step === 2 && (
        <>
          <p style={{ fontSize: 16, maxWidth: "62ch" }}>
            Both solvers were evaluated on the 40 instances of Beasley's OR-Library, a standard benchmark for this problem, and compared against exact MILP optima where available.
          </p>
          <div className="row">
            <Card n="STRENGTH" title="Small and medium instances">Both approaches reach the best known solution in nearly all runs.</Card>
            <Card n="LIMITATION" title="Largest instances">On the 100-facility, 1000-customer instances the hybrid currently underperforms the classical GA.</Card>
          </div>
        </>
      )}
      {step === 3 && (
        <>
          <p style={{ fontSize: 16 }}>Suggested reading order:</p>
          <table style={{ maxWidth: 620 }}>
            <tbody>
              <tr><td>Lesson 2</td><td style={{ textAlign: "left" }}>Problem formulation: toggle facilities and observe the cost trade-off.</td></tr>
              <tr><td>Lesson 3</td><td style={{ textAlign: "left" }}>Baseline methods: greedy heuristic and exact MILP.</td></tr>
              <tr><td>Lesson 4</td><td style={{ textAlign: "left" }}>Classical genetic algorithm, with a live run on cap71.</td></tr>
              <tr><td>Lesson 5</td><td style={{ textAlign: "left" }}>Hybrid ML-GA: the surrogate-assisted pipeline.</td></tr>
              <tr><td>Lessons 6-7</td><td style={{ textAlign: "left" }}>Full benchmark results and limitations.</td></tr>
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}

/* ============ 1. The problem ============ */
function Problem({ step }) {
  const [mask, setMask] = useState(allOpen());
  const ev = useMemo(() => evaluate(mask), [mask]);
  return (
    <>
      <div className="chips">
        <Chip label="Facilities" value={D.m} />
        <Chip label="Customers" value={D.n} />
        <Chip label="Open" value={sum(mask)} />
        <Chip label="Fixed cost" value={step >= 1 ? money(ev.fixedCost) : "—"} />
        <Chip label="Transport" value={step >= 3 ? money(ev.transport) : "—"} />
        <Chip label="Total" value={step >= 3 ? money(ev.total) : "—"} hot />
      </div>
      <div className="stage">
        <FacilityMap
          open={mask}
          flows={ev.flows}
          showFlows={step >= 3}
          used={step >= 2 ? ev.used : null}
          onToggle={step >= 2 ? (i) => setMask((m) => m.map((v, k) => (k === i ? 1 - v : v))) : undefined}
          focus={step === 1 ? 0 : null}
        />
        {step >= 2 && (
          <div style={{ marginTop: 8 }}>
            {ev.feasible ? <span className="tag g">feasible</span> : <span className="tag o">infeasible: {Math.round(ev.unmet)} demand unserved</span>}
            <span style={{ marginLeft: 10, color: "var(--muted)", fontSize: 12 }}>Click facilities to open or close them. Known optimal cost: {money(D.optimal)}</span>
          </div>
        )}
      </div>
    </>
  );
}

/* ============ 2. Baselines ============ */
function Baselines({ step }) {
  const order = useMemo(greedyOrder, []);
  const firstFeasible = useMemo(() => order.findIndex((_, t) => evaluate(maskOf(order.slice(0, t + 1))).feasible) + 1, [order]);
  const k = step === 0 ? 0 : step === 1 ? 3 : firstFeasible;
  const open = maskOf(order.slice(0, k));
  const ev = evaluate(open);
  const gap = ((ev.total - D.optimal) / D.optimal) * 100;
  if (step === 3) {
    return (
      <div className="stage">
        <table>
          <thead><tr><th>Instance</th><th>Best known</th><th>MILP gap</th><th>Greedy gap</th><th>GA gap</th><th>MILP status</th></tr></thead>
          <tbody>
            {results.large.map((r) => (
              <tr key={r.dataset}>
                <td>{r.dataset}</td><td>{money(r.ground_truth)}</td><td>{pct(r.milp_gap, 1)}</td>
                <td className="bad">{pct(r.greedy_gap, 1)}</td><td className={r.ga_gap < 0 ? "good" : ""}>{pct(r.ga_gap, 1)}</td>
                <td style={{ fontSize: 11 }}>{String(r.milp_status).startsWith("Optimal") ? "proved optimal" : "time limit"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="note">On the 100×1000 instances CBC hits its time limit with a feasible but unproven answer. Greedy is instant but far from the best known cost.</div>
      </div>
    );
  }
  return (
    <>
      <div className="chips">
        <Chip label="Facilities opened" value={k} />
        <Chip label="Feasible" value={k === 0 ? "—" : ev.feasible ? "yes" : "no"} />
        <Chip label="Total cost" value={k && ev.feasible ? money(ev.total) : "—"} hot />
        <Chip label="Gap to optimal" value={k && ev.feasible ? pct(gap, 1) : "—"} />
      </div>
      <div className="stage">
        <FacilityMap open={open} flows={ev.flows} used={ev.used} />
        <div style={{ marginTop: 8, fontSize: 12, color: "var(--muted)" }}>
          Opening order (lowest fixed cost per capacity first): {order.slice(0, 8).map((i) => i + 1).join(" → ")} …
        </div>
      </div>
    </>
  );
}

/* ============ 3. Classical GA ============ */
function ClassicalGA({ step }) {
  const hist = useMemo(() => runGA({ pop: 30, gens: 40 }), []);
  const [g, setG] = useState(0);
  const [play, setPlay] = useState(false);
  useEffect(() => {
    if (!play) return;
    const t = setInterval(() => setG((v) => { if (v >= 40) { setPlay(false); return v; } return v + 1; }), 160);
    return () => clearInterval(t);
  }, [play]);
  const A = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1];
  const B = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1];
  const child = A.map((v, i) => (i >= 5 && i <= 10 ? B[i] : v));
  const mut = child.slice(); mut[2] ^= 1;
  if (step <= 4) {
    const eA = evaluate(A);
    return (
      <div className="stage">
        {step === 0 && <><p>One <b>chromosome</b> is 16 bits, one per facility. 1 means open, 0 means closed.</p><Chromosome label="Individual" bits={A} /><p style={{ color: "var(--muted)" }}>A population of 30 to 120 such strings is created at random.</p></>}
        {step === 1 && <><Chromosome label="Individual" bits={A} /><div className="chips"><Chip label="Fixed" value={money(eA.fixedCost)} /><Chip label="Transport" value={money(eA.transport)} /><Chip label="Fitness" value={money(eA.total)} hot /></div></>}
        {step === 2 && <><p>Pick 3 individuals at random; the cheapest one becomes a parent.</p><Chromosome label="Parent A" bits={A} /><Chromosome label="Parent B" bits={B} /></>}
        {step === 3 && <><p>Swap the segment between two cut points (probability 0.8).</p><Chromosome label="Parent A" bits={A} mark={[5, 6, 7, 8, 9, 10]} /><Chromosome label="Parent B" bits={B} mark={[5, 6, 7, 8, 9, 10]} /><Chromosome label="Child" bits={child} /></>}
        {step === 4 && <><p>Each bit flips with probability 1/m.</p><Chromosome label="Child" bits={child} /><Chromosome label="Mutated" bits={mut} flip={2} /></>}
      </div>
    );
  }
  const h = hist[g], ev = evaluate(h.bestChrom);
  return (
    <>
      <div className="chips">
        <Chip label="Generation" value={g + "/40"} />
        <Chip label="Best cost" value={money(h.best)} hot />
        <Chip label="Gap to optimal" value={pct(((h.best - D.optimal) / D.optimal) * 100)} />
        <Chip label="Open facilities" value={sum(h.bestChrom)} />
      </div>
      <div className="stage">
        <div className="row">
          <div className="panel" style={{ flex: 1.3 }}>
            <LineChart
              series={[{ name: "best", color: "var(--green-3)", values: hist.map((x) => x.best) }, { name: "mean", color: "var(--amber)", values: hist.map((x) => x.mean) }]}
              upTo={g} optimum={D.optimal} yfmt={(v) => (v / 1000).toFixed(0) + "k"} />
          </div>
          <div className="panel"><FacilityMap open={h.bestChrom} flows={ev.flows} used={ev.used} /></div>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <button className="btn" onClick={() => { if (g >= 40) setG(0); setPlay(!play); }}>{play ? "Pause" : g >= 40 ? "Replay" : "Run"}</button>
          <input type="range" min="0" max="40" value={g} onChange={(e) => { setPlay(false); setG(+e.target.value); }} style={{ flex: 1 }} />
        </div>
        <div className="note">Live demo: a small GA running in your browser on cap71 with a fast greedy assignment as fitness. The real Python/DEAP solver solves an LP for every individual.</div>
      </div>
    </>
  );
}

/* ============ 4. Hybrid ML-GA ============ */
const FLOW = [
  ["Bootstrap", "run exact LP on random individuals"],
  ["Train", "fit surrogate: bits to cost"],
  ["Warm-up", "first generations stay exact"],
  ["Predict", "surrogate scores the population"],
  ["Verify", "exact LP only if prediction beats best"],
  ["Retrain", "adopt new model only if better"],
];
function Hybrid({ step }) {
  const h = cap71.hybrid, tot = h.exact + h.surrogate;
  return (
    <>
      <div className="chips">
        <Chip label="Surrogate R²" value={h.r2.toFixed(4)} />
        <Chip label="Exact evals" value={h.exact} />
        <Chip label="Surrogate evals" value={h.surrogate} />
        <Chip label="Exact share" value={pct((h.exact / tot) * 100, 1)} hot />
      </div>
      <div className="stage">
        <div style={{ display: "flex", flexWrap: "wrap", gap: 14, justifyContent: "center", padding: "10px 0" }}>
          {FLOW.map(([t, d], i) => (
            <React.Fragment key={t}>
              <div style={{
                width: 170, padding: 12, border: "1.5px solid var(--ink)", borderRadius: 12, boxShadow: "var(--pop)", transition: ".25s",
                background: i === step ? "var(--green)" : i < step ? "var(--mint)" : "var(--paper-2)",
                transform: i === step ? "translateY(-4px)" : "none", opacity: i > step ? 0.55 : 1,
              }}>
                <div style={{ fontSize: 11, color: "var(--muted)" }}>0{i + 1}</div>
                <b>{t}</b>
                <div style={{ fontSize: 12 }}>{d}</div>
              </div>
              {i < FLOW.length - 1 && <span style={{ alignSelf: "center", fontSize: 20 }}>→</span>}
            </React.Fragment>
          ))}
        </div>
        <div style={{ marginTop: 18 }}>
          <div style={{ fontSize: 12, color: "var(--muted)", marginBottom: 4 }}>Where each fitness evaluation went (cap71, per run)</div>
          <div style={{ display: "flex", height: 30, border: "1.5px solid var(--ink)", borderRadius: 8, overflow: "hidden" }}>
            <div style={{ width: (h.exact / tot) * 100 + "%", background: "var(--rust)", color: "#fff", fontSize: 11, display: "grid", placeItems: "center" }}>exact</div>
            <div style={{ flex: 1, background: "var(--green)", display: "grid", placeItems: "center", fontSize: 11 }}>surrogate (cheap)</div>
          </div>
        </div>
      </div>
    </>
  );
}

/* ============ 5. Benchmarks ============ */
function Benchmarks() {
  const [sort, setSort] = useState({ k: "name", d: 1 });
  const [sel, setSel] = useState("cap71");
  const rows = useMemo(() => {
    const get = (r) => ({ name: r.name, opt: r.optimal, cg: r.classical.avgGap, hg: r.hybrid.avgGap, ct: r.classical.time, ht: r.hybrid.time, r2: r.hybrid.r2 })[sort.k];
    return [...results.instances].sort((a, b) => (get(a) > get(b) ? 1 : get(a) < get(b) ? -1 : 0) * sort.d);
  }, [sort]);
  const s = results.instances.find((x) => x.name === sel);
  const th = (k, t) => <th onClick={() => setSort((o) => ({ k, d: o.k === k ? -o.d : 1 }))}>{t}{sort.k === k ? (sort.d > 0 ? " ▲" : " ▼") : ""}</th>;
  const I = results.instances;
  return (
    <>
      <div className="chips">
        <Chip label="Instances" value={I.length} />
        <Chip label="Hybrid hit optimum" value={I.filter((r) => r.hybrid.bestGap < 1e-6).length} />
        <Chip label="Classical hit optimum" value={I.filter((r) => r.classical.bestGap < 1e-6).length} />
        <Chip label="Hybrid better avg" value={I.filter((r) => r.hybrid.avgGap < r.classical.avgGap - 1e-6).length} hot />
      </div>
      <div className="row">
        <div className="stage panel" style={{ flex: 2, maxHeight: 520, overflow: "auto", padding: 0 }}>
          <table>
            <thead><tr>{th("name", "Instance")}{th("opt", "Optimal")}{th("cg", "Classical gap")}{th("hg", "Hybrid gap")}{th("ct", "Classical t")}{th("ht", "Hybrid t")}{th("r2", "R²")}</tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name} className={sel === r.name ? "sel" : ""} onClick={() => setSel(r.name)}>
                  <td>{r.name}</td><td>{money(r.optimal)}</td><td>{pct(r.classical.avgGap)}</td>
                  <td className={r.hybrid.avgGap > r.classical.avgGap + 1e-6 ? "bad" : ""}>{pct(r.hybrid.avgGap)}</td>
                  <td>{secs(r.classical.time)}</td><td>{secs(r.hybrid.time)}</td><td>{r.hybrid.r2.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="stage panel" style={{ minWidth: 260 }}>
          <b>{s.name}</b>
          <p style={{ color: "var(--muted)", margin: "4px 0 12px" }}>Optimal {money(s.optimal)}</p>
          {[["Classical GA", s.classical, "var(--amber)"], ["Hybrid ML-GA", s.hybrid, "var(--green-2)"]].map(([n, v, c]) => (
            <div key={n} style={{ marginBottom: 12 }}>
              <span className="tag" style={{ background: c }}>{n}</span>
              <div style={{ fontSize: 12, marginTop: 6 }}>best {money(v.best)}<br />avg {money(v.avg)}<br />avg gap {pct(v.avgGap, 3)} · {secs(v.time)}</div>
            </div>
          ))}
          <div style={{ fontSize: 12 }}>Hybrid exact / surrogate evals: {s.hybrid.exact} / {s.hybrid.surrogate}</div>
        </div>
      </div>
    </>
  );
}

/* ============ 6. Limits ============ */
function Limits({ step }) {
  const big = results.instances.filter((r) => /^cap[abc]/.test(r.name));
  const I = results.instances;
  return (
    <div className="stage">
      {step === 0 && (
        <>
          <p>On the small and medium OR-Library instances both GAs reach (or nearly reach) the optimum.</p>
          <div className="chips">
            <Chip label="Classical median avg gap" value={pct(median(I.map((r) => r.classical.avgGap)), 3)} />
            <Chip label="Hybrid median avg gap" value={pct(median(I.map((r) => r.hybrid.avgGap)), 3)} hot />
          </div>
        </>
      )}
      {step === 1 && (
        <>
          <p>The largest instances (100 facilities × 1000 customers):</p>
          <table>
            <thead><tr><th>Instance</th><th>Classical gap</th><th>Hybrid gap</th></tr></thead>
            <tbody>{big.map((r) => (<tr key={r.name}><td>{r.name}</td><td>{pct(r.classical.avgGap)}</td><td className={r.hybrid.avgGap > r.classical.avgGap ? "bad" : ""}>{pct(r.hybrid.avgGap)}</td></tr>))}</tbody>
          </table>
        </>
      )}
      {step === 2 && (
        <>
          <p>On these instances CBC cannot prove optimality, so a "gap" is measured against a best-known value rather than a certified optimum.</p>
          <div className="note">Honest summary: the surrogate cuts exact evaluations, but on the largest instances the Hybrid currently trails the Classical GA.</div>
        </>
      )}
    </div>
  );
}

export const SECTIONS = ["Foundations", "Search", "Evidence"];
export const LESSONS = [
  { id: "intro", sec: 0, title: "Research Overview", sub: "Problem, method and findings of the Hybrid ML-GA solver for the Capacitated Facility Location Problem.", level: "Overview", time: 3, C: Intro,
    steps: [
      ["Problem statement", "Select which facilities to open and assign customers to them at minimum total cost, under capacity constraints."],
      ["Problem, method, contribution", "A genetic algorithm explores facility configurations; a learned surrogate reduces the number of exact cost evaluations."],
      ["Evaluation", "Benchmarked on 40 OR-Library instances. Results are strong on small and medium instances, with a documented limitation on the largest."],
      ["Reading guide", "Proceed with Next, or jump to the benchmark lesson for the full results."],
    ] },
  { id: "problem", sec: 0, title: "The Facility Location Problem", sub: "Which warehouses do we open, and who do they serve? Click facilities and watch the bill change.", level: "Beginner", time: 8, C: Problem,
    steps: [
      ["Meet the cast", "16 candidate facilities (squares) and 50 customers (dots): OR-Library instance cap71. Map positions are schematic."],
      ["Opening costs money", "Each facility has a fixed opening cost. Open too many and you overspend."],
      ["Open and close", "Toggle facilities. Capacity limits how much each can serve; the fill level shows inside each square."],
      ["Transport adds up", "Every customer is served from open facilities. Total = fixed + transport. The goal is to minimize it."],
    ] },
  { id: "baselines", sec: 0, title: "Baselines: Greedy vs MILP", sub: "A quick heuristic and an exact solver, and why neither is enough.", level: "Beginner", time: 6, C: Baselines,
    steps: [
      ["Start with nothing", "Greedy ranks facilities by fixed cost per unit of capacity."],
      ["Open the cheapest few", "It opens facilities in that order..."],
      ["...until demand is covered", "It stops once the solution is feasible. Fast, but it ignores transport cost, so it is not optimal."],
      ["Exact is slow at scale", "MILP (PuLP + CBC) proves optimality on small cases but hits time limits on 100x1000 instances."],
    ] },
  { id: "ga", sec: 1, title: "Classical Genetic Algorithm", sub: "Evolve open/closed bit-strings with selection, crossover and mutation.", level: "Intermediate", time: 10, C: ClassicalGA,
    steps: [
      ["Chromosome", "A candidate solution is a bit-string of facility open/closed flags."],
      ["Fitness", "Fitness is the fixed cost plus the best transport assignment for that mask."],
      ["Selection", "Tournament of 3: better individuals get to reproduce."],
      ["Crossover", "Two-point crossover recombines two parents."],
      ["Mutation", "Random bit flips keep the population diverse."],
      ["Run it", "Press Run to evolve 40 generations live. Elitism keeps the best solution."],
    ] },
  { id: "hybrid", sec: 1, title: "Hybrid ML-GA", sub: "Replace most expensive fitness evaluations with a learned surrogate model.", level: "Advanced", time: 12, C: Hybrid,
    steps: [
      ["Bootstrap its own data", "No pre-trained model needed: run exact evaluations on random individuals first."],
      ["Train the surrogate", "A regressor learns bit-string to cost (R² is shown above for cap71; it varies by instance)."],
      ["Warm-up generations", "Early generations stay exact until the model is trustworthy."],
      ["Predict, don't solve", "The surrogate scores the whole population almost instantly."],
      ["Verify only promising ones", "The exact LP runs only when a prediction suggests a new best solution."],
      ["Quality-gated retraining", "A retrained model replaces the old one only if it is better."],
    ] },
  { id: "bench", sec: 2, title: "40-Instance Benchmark", sub: "Sort, select and compare Classical GA and Hybrid ML-GA on every OR-Library instance.", level: "Results", time: 10, C: Benchmarks,
    steps: [["Explore", "Click a column to sort and a row for details. Red marks where Hybrid is worse than Classical."]] },
  { id: "limits", sec: 2, title: "Honest Limits", sub: "What the results do and do not show.", level: "Results", time: 4, C: Limits,
    steps: [["Where it works", "Small and medium instances."], ["Where it struggles", "The largest instances."], ["How to read the gaps", "Caveat about unproven optima."]] },
];
