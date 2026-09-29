const API = "http://127.0.0.1:8000";
const out = document.getElementById("out");

function line(tag, text, cls) { const e = document.createElement(tag); e.textContent = text; if (cls) e.className = cls; return e; }

async function scan() {
  out.replaceChildren(line("p", "Analyzing…", "muted"));
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = tab && tab.url;
  document.getElementById("url").textContent = url || "";
  if (!url || !/^https?:\/\//i.test(url)) return out.replaceChildren(line("p", "This page type (e.g. chrome://) cannot be analyzed.", "muted"));
  try {
    const res = await fetch(`${API}/api/v1/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) });
    const j = await res.json();
    if (!res.ok) throw new Error(typeof j.detail === "string" ? j.detail : "Request rejected.");
    out.replaceChildren(line("div", `${j.classification} · ${j.risk_percentage}%`, `cls ${j.classification}`));
    const ul = document.createElement("ul");
    (j.reasons.length ? j.reasons : ["No risk-increasing signals found."]).forEach((r) => ul.append(line("li", r)));
    out.append(ul);
    chrome.runtime.sendMessage({ type: "badge", tabId: tab.id, classification: j.classification });
  } catch (e) {
    out.replaceChildren(line("p", e instanceof TypeError ? "SnapShield backend is offline. Start it with: python run.py" : e.message, "err"));
  }
}
document.getElementById("retry").onclick = scan;
scan();
