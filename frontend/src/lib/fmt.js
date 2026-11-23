export const money = (v) => (v == null ? "—" : "$" + Math.round(v).toLocaleString("en-US"));
export const pct = (v, d = 2) => (v == null ? "—" : (Math.abs(v) < 1e-9 ? 0 : v).toFixed(d) + "%");
export const secs = (v) => (v == null ? "—" : v < 100 ? v.toFixed(1) + "s" : Math.round(v) + "s");
