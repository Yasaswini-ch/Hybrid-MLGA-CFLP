import React, { useEffect, useState } from "react";
import { LESSONS, SECTIONS } from "./lessons/index.jsx";
import { Caption } from "./components.jsx";
import Hero from "./hero.jsx";

const read = () => location.hash.replace("#/", "") || LESSONS[0].id;
const useHash = () => {
  const [h, setH] = useState(read());
  useEffect(() => {
    const f = () => setH(read());
    addEventListener("hashchange", f);
    return () => removeEventListener("hashchange", f);
  }, []);
  return h;
};

export default function App() {
  const id = useHash();
  const idx = Math.max(0, LESSONS.findIndex((l) => l.id === id));
  const L = LESSONS[idx];
  const [rawStep, setStep] = useState(0);
  const [dark, setDark] = useState(false);
  const [menu, setMenu] = useState(false);
  useEffect(() => { setStep(0); scrollTo(0, 0); setMenu(false); }, [id]);
  useEffect(() => { document.documentElement.dataset.theme = dark ? "dark" : "light"; }, [dark]);
  const step = Math.min(rawStep, L.steps.length - 1);
  const last = step >= L.steps.length - 1;
  const next = LESSONS[idx + 1];
  const go = () => (last ? next && (location.hash = "#/" + next.id) : setStep(step + 1));
  const back = () => (step > 0 ? setStep(step - 1) : idx > 0 && (location.hash = "#/" + LESSONS[idx - 1].id));
  const [t, d] = L.steps[step];
  return (
    <div className="app">
      <header className="top">
        <button className="icon-btn" onClick={() => setMenu(!menu)} aria-label="menu">☰</button>
        <div className="brand">▦ Hybrid <span className="hl">ML-GA</span></div>
        <div className="grow" />
        <div className="credit">Capacitated Facility<br />Location Problem</div>
        <button className="icon-btn" onClick={() => setDark(!dark)} aria-label="theme">{dark ? "☀" : "☾"}</button>
      </header>
      <div className="layout">
        <nav className={"side" + (menu ? " open" : "")}>
          {SECTIONS.map((s, si) => (
            <div key={s}>
              <h4>{String(si + 1).padStart(2, "0")}. {s}</h4>
              {LESSONS.map((l, i) => l.sec === si && (
                <a key={l.id} href={"#/" + l.id} className={i === idx ? "on" : ""}><span className="num">{i + 1}</span>{l.title}</a>
              ))}
            </div>
          ))}
        </nav>
        <main className="main">
          {L.id === "intro" && <Hero />}
          <div className="crumb">{String(L.sec + 1).padStart(2, "0")}. {SECTIONS[L.sec]} / Section {idx + 1} of {LESSONS.length}</div>
          <h1>{L.title}</h1>
          <div className="pills"><span className="pill g">{SECTIONS[L.sec]}</span><span className="pill v">{L.steps.length} parts</span></div>
          <p className="sub">{L.sub}</p>
          <L.C step={step} />
          <Caption title={t}>{d}</Caption>
          <div className="steps">
            {L.steps.map((_, i) => (<button key={i} className={"dot" + (i < step ? " done" : i === step ? " on" : "")} onClick={() => setStep(i)} aria-label={"step " + (i + 1)} />))}
          </div>
        </main>
      </div>
      <div className="bar">
        <div className="in">
          <button className="btn" onClick={back} disabled={idx === 0 && step === 0}>← Back</button>
          <button className="btn next" onClick={go} disabled={last && !next}>
            <span>{last ? (next ? "Next section: " + next.title : "Done") : "Next: " + L.steps[step + 1][0]}</span><span>→</span>
          </button>
        </div>
      </div>
    </div>
  );
}
