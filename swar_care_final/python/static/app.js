// app.js — core wiring: websocket telemetry, tabs, control deck, status banner,
// recording timer, storage metrics panel. terminal.js / scope.js / veena.js
// each listen for the "swarcare:telemetry" event this file dispatches.

window.SwarCare = window.SwarCare || {};

(function () {
  const SC = window.SwarCare;
  SC.latest = { status: null, audio: null, piezo: null };
  SC.timeSynced = false;

  // ---------------------------------------------------------------------
  // Time-sync — send device clock to the server on first load
  // ---------------------------------------------------------------------
  (async function syncDeviceTime() {
    const MAX_RETRIES = 3;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      try {
        const res = await fetch("/api/sync_time", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ epoch_ms: Date.now() }),
        });
        if (res.ok) {
          const data = await res.json();
          SC.timeSynced = true;
          document.dispatchEvent(new CustomEvent("swarcare:timesync", { detail: data }));
          SC.toast("🕐 Time synced → " + data.server_time_ist);
          return;
        }
      } catch (_) { /* retry */ }
      await new Promise((r) => setTimeout(r, 800 * attempt));
    }
    SC.toast("⚠️ Time sync failed — timestamps may be inaccurate");
  })();

  // ---------------------------------------------------------------------
  // Toast
  // ---------------------------------------------------------------------
  let toastEl = null;
  let toastTimer = null;
  SC.toast = function (msg) {
    if (!toastEl) {
      toastEl = document.createElement("div");
      toastEl.id = "toast";
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = msg;
    toastEl.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove("show"), 2200);
  };

  // ---------------------------------------------------------------------
  // Tabs
  // ---------------------------------------------------------------------
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "tab-records" && window.SwarCareRecords) {
        window.SwarCareRecords.refresh();
      }
    });
  });

  // ---------------------------------------------------------------------
  // Control deck
  // ---------------------------------------------------------------------
  const btnStart = document.getElementById("btn-start");
  const btnPause = document.getElementById("btn-pause");
  const btnStop = document.getElementById("btn-stop");

  async function post(path) {
    const res = await fetch(path, { method: "POST" });
    if (!res.ok) throw new Error("Request failed: " + path);
    return res.json();
  }

  btnStart.addEventListener("click", async () => {
    btnStart.disabled = true;
    try { await post("/api/record/start"); } catch (e) { SC.toast("⚠️ Could not start recording"); }
  });
  btnPause.addEventListener("click", async () => {
    btnPause.disabled = true;
    try { await post("/api/record/pause"); } catch (e) { SC.toast("⚠️ Could not pause"); }
  });
  btnStop.addEventListener("click", async () => {
    btnStop.disabled = true;
    try { await post("/api/record/stop"); } catch (e) { SC.toast("⚠️ Could not stop"); }
  });

  function updateControlDeck(state) {
    btnStart.textContent = state === "PAUSED" ? "▶ RESUME" : "▶ START RECORDING";
    btnStart.disabled = state === "RECORDING" || state === "STOPPING";
    btnPause.disabled = state !== "RECORDING";
    btnStop.disabled = !(state === "RECORDING" || state === "PAUSED");
  }

  // ---------------------------------------------------------------------
  // Status banner
  // ---------------------------------------------------------------------
  const banner = document.getElementById("status-banner");
  const BANNER_MAP = {
    STOPPED: ["banner-error", "System Status: <strong>STOPPED / IDLE</strong> ⏹️"],
    RECORDING: ["banner-success", "System Status: <strong>RECORDING SENSORS LIVE</strong> ▶️"],
    PAUSED: ["banner-warning", "System Status: <strong>RECORDING PAUSED</strong> ⏸️"],
    STOPPING: ["banner-info", "System Status: <strong>WRITING DATA STREAMS GRACEFULLY... PLEASE WAIT</strong> ⏳"],
  };
  function updateStatusBanner(state) {
    const [cls, html] = BANNER_MAP[state] || BANNER_MAP.STOPPED;
    banner.className = "banner " + cls;
    banner.innerHTML = html;
  }

  // ---------------------------------------------------------------------
  // Recording timer (ported from the old timer_static_html iframe script)
  // ---------------------------------------------------------------------
  const timerBox = document.getElementById("timer-box");
  const timerLabel = document.getElementById("timer-label");
  const timerValue = document.getElementById("timer-value");
  let anchorStartMs = null;
  let maxDisplaySec = 0.0;

  function fmt(sec) {
    sec = Math.max(0, Math.floor(sec));
    const hh = String(Math.floor(sec / 3600)).padStart(2, "0");
    const mm = String(Math.floor((sec % 3600) / 60)).padStart(2, "0");
    const ss = String(sec % 60).padStart(2, "0");
    return hh + ":" + mm + ":" + ss;
  }

  function styleVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function updateTimer(state, serverElapsed) {
    if (state === "PAUSED") {
      anchorStartMs = null;
      timerBox.style.borderColor = styleVar("--accent-amber");
      timerLabel.style.color = styleVar("--accent-amber");
      timerLabel.textContent = "⏱️ PAUSED AT";
      timerValue.textContent = fmt(serverElapsed);
    } else if (state === "STOPPING") {
      anchorStartMs = null;
      timerBox.style.borderColor = styleVar("--accent-cyan");
      timerLabel.style.color = styleVar("--accent-cyan");
      timerLabel.textContent = "⏱️ SAVING RECORDING...";
      timerValue.textContent = fmt(serverElapsed);
    } else if (state === "RECORDING") {
      timerBox.style.borderColor = styleVar("--accent-green");
      timerLabel.style.color = styleVar("--accent-green");
      timerLabel.textContent = "⏱️ RECORDING TIME";

      const now = performance.now();
      if (anchorStartMs === null) {
        anchorStartMs = now - serverElapsed * 1000;
        maxDisplaySec = serverElapsed;
      }
      let calculatedSec = (now - anchorStartMs) / 1000.0;
      if (serverElapsed > calculatedSec + 1.5) {
        anchorStartMs = now - serverElapsed * 1000;
        calculatedSec = serverElapsed;
      }
      if (calculatedSec > maxDisplaySec) maxDisplaySec = calculatedSec;
      timerValue.textContent = fmt(maxDisplaySec);
    } else {
      anchorStartMs = null;
      maxDisplaySec = 0.0;
      timerBox.style.borderColor = styleVar("--border-subtle");
      timerLabel.style.color = styleVar("--text-secondary");
      timerLabel.textContent = "⏱️ RECORDING TIME";
      timerValue.textContent = fmt(0);
    }
  }

  function animTimer() {
    if (SC.latest.status) {
      updateTimer(SC.latest.status.state, SC.latest.status.elapsed_s);
    }
    requestAnimationFrame(animTimer);
  }
  requestAnimationFrame(animTimer);

  // ---------------------------------------------------------------------
  // Storage metrics panel
  // ---------------------------------------------------------------------
  function updateStorageMetrics(status) {
    const pSamples = status.piezo_samples;
    const pLines = pSamples > 0 ? pSamples + 1 : 0;
    const pTime = pSamples / status.piezo_sample_rate_hz;
    const aSamples = status.audio_samples;
    const aTime = aSamples / status.audio_sample_rate_hz;

    document.getElementById("storage-piezo-samples").textContent = pSamples.toLocaleString();
    document.getElementById("storage-piezo-lines").textContent = pLines.toLocaleString();
    document.getElementById("storage-piezo-duration").textContent = pTime.toFixed(2) + " s";
    document.getElementById("storage-audio-samples").textContent = aSamples.toLocaleString();
    document.getElementById("storage-audio-wav").textContent = aSamples.toLocaleString();
    document.getElementById("storage-audio-duration").textContent = aTime.toFixed(2) + " s";
  }

  // ---------------------------------------------------------------------
  // WebSocket telemetry
  // ---------------------------------------------------------------------
  let prevState = null;

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(proto + "//" + location.host + "/ws/telemetry");

    ws.onmessage = (evt) => {
      let payload;
      try { payload = JSON.parse(evt.data); } catch (e) { return; }

      SC.latest.status = payload.status;
      SC.latest.audio = payload.audio;
      SC.latest.piezo = payload.piezo;

      const state = payload.status.state;
      updateControlDeck(state);
      updateStatusBanner(state);
      updateStorageMetrics(payload.status);

      // Keep the time-sync indicator in the header up to date
      const badge = document.getElementById("time-sync-badge");
      if (badge) {
        if (payload.status.time_synced) {
          badge.textContent = "🕐 Time synced";
          badge.className = "sync-badge synced";
        } else {
          badge.textContent = "⚠️ Time not synced";
          badge.className = "sync-badge unsynced";
        }
      }

      if (prevState === "STOPPING" && state === "STOPPED") {
        SC.toast("🎉 Recording saved and verified successfully!");
        if (document.getElementById("tab-records").classList.contains("active") && window.SwarCareRecords) {
          window.SwarCareRecords.refresh();
        }
      }
      prevState = state;

      document.dispatchEvent(new CustomEvent("swarcare:telemetry", { detail: payload }));
    };

    ws.onclose = () => setTimeout(connect, 1000);
    ws.onerror = () => ws.close();
  }
  connect();
})();
