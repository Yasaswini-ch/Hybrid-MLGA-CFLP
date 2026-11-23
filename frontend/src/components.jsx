import React, { useMemo } from "react";
import { D, rng } from "./lib/cflp.js";

// ---------- Schematic map (OR-Library has no coordinates, so positions are generated) ----------
const W = 640, H = 400;
export function layout() {
  const r = rng(11), cols = 4;
  const fac = Array.from({ length: D.m }, (_, i) => ({ x: 90 + (i % cols) * 153 + (r() - .5) * 20, y: 60 + Math.floor(i / cols) * 92 + (r() - .5) * 14 }));
  const cus = D.cost.map((row) => {
    let b = 0; row.forEach((c, i) => { if (c < row[b]) b = i; });
    const a = r() * Math.PI * 2, d = 22 + r() * 34;
    return { x: Math.min(W - 14, Math.max(14, fac[b].x + Math.cos(a) * d * 1.3)), y: Math.min(H - 14, Math.max(14, fac[b].y + Math.sin(a) * d * .9)) };
  });
  return { fac, cus };
}

export function FacilityMap({ open, flows = [], showCustomers = true, showFlows = true, onToggle, used, focus }) {
  const { fac, cus } = useMemo(layout, []);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block", maxHeight: 420 }}>
      {showFlows && flows.map(([j, i, f], k) => (
        <line key={k} x1={cus[j].x} y1={cus[j].y} x2={fac[i].x} y2={fac[i].y} stroke="var(--green-3)" strokeWidth={0.6 + f * 1.4} opacity=".55" />
      ))}
      {showCustomers && cus.map((c, j) => (
        <circle key={j} cx={c.x} cy={c.y} r={3 + Math.min(4, D.dem[j] / 250)} fill="var(--blue)" stroke="var(--ink)" strokeWidth="1" />
      ))}
      {fac.map((f, i) => {
        const on = open[i], dim = focus != null && focus !== i;
        const fill = used ? Math.min(1, used[i] / D.cap[i]) : 0;
        return (
          <g key={i} className="fac" onClick={() => onToggle && onToggle(i)} opacity={dim ? .35 : 1}>
            <rect x={f.x - 17} y={f.y - 17} width="34" height="34" rx="8" fill={on ? "var(--green)" : "var(--paper-2)"} stroke="var(--ink)" strokeWidth="1.5" />
            {on && used && <rect x={f.x - 17} y={f.y + 17 - 34 * fill} width="34" height={34 * fill} rx="6" fill="var(--green-2)" opacity=".8" />}
            <text x={f.x} y={f.y + 4} textAnchor="middle" fontSize="11" fontWeight="700" fill="var(--ink)">{i + 1}</text>
            {!on && <line x1={f.x - 12} y1={f.y + 12} x2={f.x + 12} y2={f.y - 12} stroke="var(--muted-2)" strokeWidth="1.2" />}
          </g>
        );
      })}
    </svg>
  );
}

// ---------- Line chart ----------
export function LineChart({ series, w = 620, h = 300, xlabel = "generation", yfmt = (v) => v, optimum: refLine, upTo }) {
  const pad = { l: 64, r: 16, t: 14, b: 34 };
  const all = series.flatMap((s) => s.values), n = Math.max(...series.map((s) => s.values.length));
  let lo = Math.min(...all, refLine ?? Infinity), hi = Math.max(...all);
  const span = hi - lo || 1; lo -= span * .05; hi += span * .05;
  const X = (i) => pad.l + (i / Math.max(1, n - 1)) * (w - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
  const ticks = [0, .25, .5, .75, 1].map((t) => lo + t * (hi - lo));
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%">
      {ticks.map((t, i) => (<g key={i}><line x1={pad.l} x2={w - pad.r} y1={Y(t)} y2={Y(t)} stroke="var(--line)" /><text x={pad.l - 8} y={Y(t) + 4} textAnchor="end" fontSize="10" fill="var(--muted)">{yfmt(t)}</text></g>))}
      {refLine != null && <g><line x1={pad.l} x2={w - pad.r} y1={Y(refLine)} y2={Y(refLine)} stroke="var(--rust)" strokeDasharray="5 4" strokeWidth="1.5" /><text x={w - pad.r} y={Y(refLine) - 5} textAnchor="end" fontSize="10" fill="var(--rust)">MILP optimum</text></g>}
      {series.map((s) => {
        const vals = upTo != null ? s.values.slice(0, upTo + 1) : s.values;
        return (<g key={s.name}><polyline fill="none" stroke={s.color} strokeWidth="2.4" strokeLinejoin="round" points={vals.map((v, i) => `${X(i)},${Y(v)}`).join(" ")} />
          {vals.length > 0 && <circle cx={X(vals.length - 1)} cy={Y(vals[vals.length - 1])} r="4.5" fill={s.color} stroke="var(--ink)" strokeWidth="1.5" />}</g>);
      })}
      <text x={(w + pad.l) / 2} y={h - 6} textAnchor="middle" fontSize="11" fill="var(--muted)">{xlabel}</text>
      <g transform={`translate(${pad.l + 10},${pad.t + 6})`}>{series.map((s, i) => (<g key={s.name} transform={`translate(${i * 130},0)`}><rect width="12" height="12" rx="3" fill={s.color} stroke="var(--ink)" /><text x="18" y="10" fontSize="11" fill="var(--ink)">{s.name}</text></g>))}</g>
    </svg>
  );
}

export const Chip = ({ label, value, hot }) => (<div className={"chip" + (hot ? " hot" : "")}><small>{label}</small><b>{value}</b></div>);
export const Caption = ({ title, children }) => (<div className="caption"><b>{title}</b>{children}</div>);

// ---------- Chromosome strip ----------
export function Chromosome({ bits, mark, label, flip }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "6px 0" }}>
      <span style={{ width: 70, fontSize: 11, color: "var(--muted)" }}>{label}</span>
      <div style={{ display: "flex", gap: 3 }}>
        {bits.map((b, i) => (
          <div key={i} style={{ width: 26, height: 30, display: "grid", placeItems: "center", border: "1.5px solid var(--ink)", borderRadius: 6, fontWeight: 700,
            background: flip === i ? "var(--orange)" : b ? "var(--green)" : "var(--paper-2)", outline: mark && mark.includes(i) ? "2.5px solid var(--rust)" : "none" }}>{b}</div>
        ))}
      </div>
    </div>
  );
}
