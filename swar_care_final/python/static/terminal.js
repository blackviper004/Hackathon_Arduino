// terminal.js — Vibration Serial Monitor (Piezo Sensor)
(function () {
  const box = document.getElementById("terminal-box");
  let lastRenderedSignature = "";
  let pRate = 2000;

  function styleVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function parseSec(txt, idx) {
    let m = txt.match(/(?:TIME|T)\s*:\s*(\d{1,2}):(\d{2}):([\d.]+)/i);
    if (m) return parseFloat(m[1]) * 3600 + parseFloat(m[2]) * 60 + parseFloat(m[3]);
    m = txt.match(/(?:TIME|T)\s*:\s*(\d{1,2}):([\d.]+)/i);
    if (m) return parseFloat(m[1]) * 60 + parseFloat(m[2]);
    m = txt.match(/(?:TIME|T|SEC|SECS)\s*:\s*([\d.]+)/i);
    if (m) return parseFloat(m[1]);
    return idx / pRate;
  }

  function renderLines(linesData, state) {
    if (state === "RECORDING") box.style.borderColor = styleVar("--accent-green");
    else if (state === "PAUSED") box.style.borderColor = styleVar("--accent-amber");
    else if (state === "STOPPING") box.style.borderColor = styleVar("--accent-cyan");
    else box.style.borderColor = styleVar("--border-subtle");

    if (!linesData || linesData.length === 0) {
      if (lastRenderedSignature !== "IDLE") {
        box.innerHTML = '<div class="idle-wrap">--- SERIAL MONITOR PIPELINE IDLE ---</div>';
        lastRenderedSignature = "IDLE";
      }
      return;
    }

    const currentSig = linesData.map((l) => (l.text || "") + (l.active ? "1" : "0")).join("|");
    if (currentSig === lastRenderedSignature) return;
    lastRenderedSignature = currentSig;

    const markerCol = styleVar("--term-marker-color");
    const markerBg = styleVar("--term-marker-bg");
    const normTextCol = styleVar("--text-primary");
    const activeTextCol = styleVar("--term-active");

    let html = "";
    let lastSecBlock = null;
    let baseTimeSec = null;

    linesData.forEach((item, idx) => {
      const txt = item.text || "";
      const isAct = item.active || false;

      const compact = txt.replace(/\s*:\s*/g, ":").replace(/\s*\|\s*/g, "|");
      const secVal = parseSec(txt, idx);
      if (baseTimeSec === null) baseTimeSec = secVal;

      const secBlock = Math.floor(secVal / 2) * 2;
      if (lastSecBlock !== null && secBlock !== lastSecBlock) {
        const relSec = Math.floor((secVal - baseTimeSec) / 2) * 2;
        html += `<div style="border-top: 1px dotted ${markerCol}; background: ${markerBg}; color: ${markerCol}; text-align: center; font-size: 9.5px; margin: 4px 0; padding: 2px 0; font-weight: bold; letter-spacing: 0.5px;">⏱️ ┈┈ 2s MARKER (+${relSec}s) ┈┈</div>`;
      }
      lastSecBlock = secBlock;

      const col = isAct ? activeTextCol : normTextCol;
      html += `<div style="color:${col}; font-weight:bold; line-height:1.35;">${escapeHtml(compact)}</div>`;
    });

    box.innerHTML = html;
    box.scrollTop = box.scrollHeight;
  }

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  document.addEventListener("swarcare:telemetry", (evt) => {
    const piezo = evt.detail.piezo;
    if (evt.detail.status && evt.detail.status.piezo_sample_rate_hz) {
      pRate = evt.detail.status.piezo_sample_rate_hz;
    }
    renderLines(piezo.lines, piezo.state);
  });
})();
