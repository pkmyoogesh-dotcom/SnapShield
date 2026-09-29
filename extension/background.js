// Sets a toolbar badge after the popup has scanned a tab. Makes no network requests itself.
const COLORS = { SAFE: "#22c55e", SUSPICIOUS: "#f59e0b", PHISHING: "#ef4444" };
chrome.runtime.onMessage.addListener((m) => {
  if (m.type !== "badge" || !COLORS[m.classification]) return;
  chrome.action.setBadgeText({ tabId: m.tabId, text: m.classification === "SAFE" ? "OK" : "!" });
  chrome.action.setBadgeBackgroundColor({ tabId: m.tabId, color: COLORS[m.classification] });
});
