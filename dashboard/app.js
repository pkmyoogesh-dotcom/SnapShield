const API = location.protocol.startsWith("http") ? location.origin : "http://127.0.0.1:8000";
const $ = (id) => document.getElementById(id);
const KEY = "snapshield_history";
const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } };

function renderHistory() {
  const ul = $("hist"); ul.replaceChildren();
  const h = load();
  if (!h.length) { const li = document.createElement("li"); li.textContent = "No scans yet."; ul.append(li); return; }
  h.forEach((e) => {
    const li = document.createElement("li");
    const u = document.createElement("span"); u.textContent = e.url;
    const c = document.createElement("span"); c.textContent = `${e.classification} ${e.pct}%`; c.className = e.classification;
    li.append(u, c); li.onclick = () => { $("url").value = e.url; };
    ul.append(li);
  });
}

function show(r) {
  $("result").hidden = false;
  const cls = $("cls"); cls.textContent = r.classification; cls.className = "badge " + r.classification;
  $("pct").textContent = r.risk_percentage + "%"; $("pct").className = r.classification;
  const fill = $("fill"); fill.parentElement.className = "bar " + r.classification; fill.style.width = r.risk_percentage + "%";
  $("engine").textContent = `Inference: ${r.inference} (local, on this device)`;
  const ul = $("reasons"); ul.replaceChildren();
  (r.reasons.length ? r.reasons : ["No risk-increasing signals found in the URL structure."]).forEach((t) => {
    const li = document.createElement("li"); li.textContent = t; ul.append(li);
  });
  const t = $("feat"); t.replaceChildren();
  Object.entries(r.features).forEach(([k, v]) => {
    const tr = document.createElement("tr"), a = document.createElement("td"), b = document.createElement("td");
    a.textContent = k.replaceAll("_", " "); b.textContent = Number.isInteger(v) ? v : v.toFixed(3);
    tr.append(a, b); t.append(tr);
  });
}

$("f").addEventListener("submit", async (e) => {
  e.preventDefault(); $("err").hidden = true; $("go").disabled = true;
  try {
    const res = await fetch(`${API}/api/v1/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: $("url").value }) });
    const j = await res.json();
    if (!res.ok) throw new Error(typeof j.detail === "string" ? j.detail : "Invalid input.");
    show(j);
    const h = load().filter((x) => x.url !== j.url); h.unshift({ url: j.url, classification: j.classification, pct: j.risk_percentage });
    localStorage.setItem(KEY, JSON.stringify(h.slice(0, 10))); renderHistory();
  } catch (err) { $("err").textContent = err.message; $("err").hidden = false; }
  $("go").disabled = false;
});
$("clear").onclick = () => { localStorage.removeItem(KEY); renderHistory(); };
fetch(`${API}/api/v1/health`).then((r) => r.json()).then((h) => { $("model").textContent = h.model_loaded ? `${h.model_type} (${h.inference_backend})` : "NOT TRAINED - run python -m ml.train"; }).catch(() => { $("model").textContent = "backend offline"; });
renderHistory();
