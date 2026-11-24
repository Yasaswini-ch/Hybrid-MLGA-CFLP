import React, { useEffect, useMemo, useState } from "react";
import { D, evaluate, runGA } from "./lib/cflp.js";
import { FacilityMap } from "./components.jsx";
import { money, pct } from "./lib/fmt.js";
import results from "./data/results.json";

const cap71 = results.instances.find((x) => x.name === "cap71");
const TOTAL_EVALS = cap71.hybrid.exact + cap71.hybrid.surrogate;
const EXACT_SHARE = cap71.hybrid.exact / TOTAL_EVALS;

const STAGES = [
  { k: "Problem", n: 16, t: "Which facilities should open?",
    d: "A network has 16 candidate sites and 50 customers. Open too many and fixed costs climb. Open too few and shipping costs climb, and capacity limits can make service impossible." },
  { k: "Search", n: 41, t: "A genetic algorithm explores configurations",
    d: "Each candidate is a string of open/closed flags. The algorithm keeps the cheapest candidates, recombines them and mutates them, generation after generation." },
  { k: "Predict", n: 34, t: "A learned model scores candidates instantly",
    d: "Pricing a candidate exactly means solving an optimisation problem. A random-forest surrogate estimates the cost in a fraction of the time, so most candidates never need the exact solve." },
  { k: "Verify", n: 26, t: "Only promising results are checked exactly",
    d: "A candidate is verified exactly only when its predicted cost beats the best found so far. The final reported solution is always re-checked, and here it matches the best known cost." },
];
const TOTAL_TICKS = STAGES.reduce((a, s) => a + s.n, 0);

export default function Hero() {
  const hist = useMemo(() => runGA({ pop: 30, gens: 40 }), []);
  const [tick, setTick] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setTick((v) => (v + 1) % TOTAL_TICKS), 150);
    return () => clearInterval(id);
  }, [playing]);

  let stage = 0, t = tick;
  while (t >= STAGES[stage].n) { t -= STAGES[stage].n; stage++; }
  const S = STAGES[stage];
  const frac = t / (S.n - 1);
  const jump = (i) => { setPlaying(false); setTick(STAGES.slice(0, i).reduce((a, s) => a + s.n, 0) + Math.floor(STAGES[i].n * 0.75)); };

  const final = hist[hist.length - 1];
  const mask = stage === 0 ? Array(D.m).fill(1) : stage === 1 ? hist[Math.min(t, hist.length - 1)].bestChrom : final.bestChrom;
  const ev = useMemo(() => evaluate(mask), [mask.join("")]);
  const cost = stage === 1 ? hist[Math.min(t, hist.length - 1)].best : stage === 0 ? ev.total : final.best;
  const scored = Math.round(frac * TOTAL_EVALS);
  const gap = ((cost - D.optimal) / D.optimal) * 100;

  return (
    <section className="hero">
      <div className="hero-copy">
        <div className="hero-kicker">Research project</div>
        <h2>Choosing where to open facilities, faster, with machine learning</h2>
        <p className="hero-lead">
          Hybrid ML-GA combines a genetic algorithm with a learned surrogate model to solve the Capacitated Facility Location Problem, and is benchmarked on 40 standard instances.
        </p>

        <div className="hero-steps" role="tablist">
          {STAGES.map((s, i) => (
            <button key={s.k} role="tab" aria-selected={i === stage} className={"hero-step" + (i === stage ? " on" : i < stage ? " done" : "")} onClick={() => jump(i)}>
              <span className="hero-num">{i + 1}</span>{s.k}
            </button>
          ))}
        </div>

        <div className="hero-card">
          <b>{S.t}</b>
          <p>{S.d}</p>
        </div>

        <div className="hero-ctl">
          <button className="btn" onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play"}</button>
          <div className="hero-prog"><div style={{ width: ((tick + 1) / TOTAL_TICKS) * 100 + "%" }} /></div>
        </div>
      </div>

      <div className="hero-viz">
        <div className="hero-map">
          <FacilityMap open={mask} flows={ev.flows} used={ev.used} />
        </div>
        <div className="hero-chips">
          {stage <= 1 && (
            <>
              <div className="chip"><small>Open facilities</small><b>{mask.reduce((a, b) => a + b, 0)} / {D.m}</b></div>
              <div className="chip hot"><small>{stage === 0 ? "Cost (all open)" : "Best cost found"}</small><b>{money(cost)}</b></div>
              <div className="chip"><small>{stage === 1 ? "Generation" : "Known optimum"}</small><b>{stage === 1 ? t + " / 40" : money(D.optimal)}</b></div>
            </>
          )}
          {stage === 2 && (
            <div className="hero-bar">
              <small>Candidates scored: {scored.toLocaleString("en-US")}</small>
              <div className="hero-split">
                <div style={{ width: EXACT_SHARE * 100 + "%", background: "var(--rust)" }}>exact {Math.round(EXACT_SHARE * 100)}%</div>
                <div style={{ flex: 1, background: "var(--green)" }}>predicted by the model {Math.round((1 - EXACT_SHARE) * 100)}%</div>
              </div>
            </div>
          )}
          {stage === 3 && (
            <>
              <div className="chip hot"><small>Verified exact cost</small><b>{money(final.best)}</b></div>
              <div className="chip"><small>Gap to best known</small><b>{pct(gap)}</b></div>
              <div className="chip"><small>Status</small><b>{gap < 0.005 ? "✓ matches optimum" : "verified"}</b></div>
            </>
          )}
        </div>
        <div className="hero-foot">Example: OR-Library instance cap71. Map positions are schematic.</div>
      </div>
    </section>
  );
}
