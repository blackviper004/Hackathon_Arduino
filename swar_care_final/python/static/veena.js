// veena.js — Veena tuning + structural diagnostics panel
// Ported from main.py's _TONIC_OPTIONS_UI / _STRING_OPTIONS /
// _STRUCTURAL_DEPTH_DESCRIPTIONS and render_unified_anomaly_monitor().
(function () {
  const TONIC_OPTIONS = {
    "A1 (55 Hz)": 55.0, "A#1 (58 Hz)": 58.27, "B1 (62 Hz)": 61.74,
    "C2 (65 Hz)": 65.41, "C#2 (69 Hz)": 69.3, "D2 (73 Hz)": 73.42,
    "D#2 (78 Hz)": 77.78, "E2 (82 Hz)": 82.41, "F2 (87 Hz)": 87.31,
    "F#2 (93 Hz)": 92.5, "G2 (98 Hz)": 98.0, "G#2 (104 Hz)": 103.83,
    "A2 (110 Hz)": 110.0, "A#2 (117 Hz)": 116.54, "B2 (123 Hz)": 123.47,
    "C3 (131 Hz)": 130.81, "C#3 (139 Hz)": 138.59, "D3 (147 Hz)": 146.83,
    "D#3 (156 Hz)": 155.56, "E3 (165 Hz)": 164.81, "F3 (175 Hz)": 174.61,
    "F#3 (185 Hz)": 185.0,
  };
  const STRING_OPTIONS = {
    "S1 — Sarani (Tara Sa, 2× Sa)": "S1",
    "S2 — Panchama (Pa, 1.5× Sa)": "S2",
    "S3 — Mandra Sa (tonic)": "S3",
    "S4 — Anumandra (lower Pa, 0.75× Sa)": "S4",
    "T1 — Chikari 1 (Sa, 4×)": "T1",
    "T2 — Chikari 2 (Pa, 6×)": "T2",
    "T3 — Chikari 3 (Sa, 8×)": "T3",
    "🔄 Auto-detect from pitch": null,
  };
  const STRUCTURAL_DEPTH_DESCRIPTIONS = {
    "-2": { title: "Silence / System Idle", summary: "No audio excitation detected. Waiting for musician to pluck a Veena string." },
    "-1": { title: "Non-Veena Sound / Human Voice", summary: "Acoustic signal analysis identified non-instrument sound (human speech, vocalization, or ambient noise). System requires genuine Saraswati Veena string resonance for diagnostic evaluation." },
    "0": { title: "Structurally Sound & Resonant", summary: "Instrument exhibits optimal acoustic resonance across the resonator (Kudam), bridge (Kudirai), and neck (Dandi). No structural damping detected." },
    "1": { title: "Structurally Sound & Resonant", summary: "Instrument exhibits optimal acoustic resonance across the resonator (Kudam), bridge (Kudirai), and neck (Dandi). No structural damping detected." },
    "2": { title: "Fret Wear / Misalignment", summary: "Bronze fret surface wear or loose fret fixing along the wax ledge (Melakku), causing non-linear buzz and fret contact impedance." },
    "3": { title: "String Corrosion / Oxidation", summary: "Surface oxidation and metal fatigue on steel/bronze core strings, altering linear mass density and harmonic purity." },
    "4": { title: "Bridge Tilt / Kudirai Asymmetry", summary: "Angular misalignment or uneven base contact of the main bridge (Kudirai) on the resonator soundboard plate." },
    "5": { title: "Kudam Crack / Resonator Shell Fracture", summary: "Structural hairline crack or wood joint separation in the Jackwood resonator shell (Kudam), causing internal cavity acoustic leakage." },
    "6": { title: "Loose Peg / Birudai Slippage", summary: "Taper pin friction failure in the tuning peg box (Birudai), causing continuous mechanical tension slippage under string load." },
    "7": { title: "String Buzz / Jiva Thread Mismatch", summary: "Improper contact angle between string and bridge curvature or damaged cotton/silk buzzing thread (Jiva/Javali)." },
    "8": { title: "Sympathetic Resonance Dampening", summary: "Acoustic damping in auxiliary drone resonators or wax wall structure, reducing sustain of unplucked resonant strings." },
    "9": { title: "Finish Degradation / Shellac Flaking", summary: "Degraded French polish / shellac coating on the wood body, affecting micro-porosity and ambient moisture protection." },
    "10": { title: "Detached Bridge / Base Separation", summary: "Partial adhesive separation between the bone/wood Kudirai base and the soundboard wood plate." },
    "11": { title: "Nut Groove Wear / Meru Slit Wear", summary: "Deepened or widened string guide grooves at the upper bridge (Meru), causing string rattling and open-string buzzing." },
  };
  const STATUS_EMOJI = {
    IN_TUNE: "✅", FLAT: "⬇️", SHARP: "⬆️", NO_PITCH: "🔇", SILENCE: "🟤", NON_VEENA: "🚫",
  };

  const tonicSelect = document.getElementById("tonic-select");
  const stringSelect = document.getElementById("string-select");

  Object.keys(TONIC_OPTIONS).forEach((k) => {
    const opt = document.createElement("option");
    opt.value = k; opt.textContent = k;
    if (k === "C3 (131 Hz)") opt.selected = true;
    tonicSelect.appendChild(opt);
  });
  Object.keys(STRING_OPTIONS).forEach((k) => {
    const opt = document.createElement("option");
    opt.value = k; opt.textContent = k;
    stringSelect.appendChild(opt);
  });

  const masterBanner = document.getElementById("veena-master-banner");
  const metricsCard = document.getElementById("veena-metrics-card");
  const depthCard = document.getElementById("structural-depth-card");
  const refGrid = document.getElementById("string-reference-grid");
  const rawJsonEl = document.getElementById("veena-raw-json");

  function bannerSet(el, cls, html) {
    el.className = "banner " + cls;
    el.innerHTML = html;
  }

  function renderReferenceTable(tonicHz) {
    const rows = [
      ["S1 Sarani", tonicHz * 2.0], ["S2 Panchama", tonicHz * 1.5],
      ["S3 Mandra Sa", tonicHz * 1.0], ["S4 Anumandra", tonicHz * 0.75],
      ["T1 Chikari", tonicHz * 4.0], ["T2 Chikari", tonicHz * 6.0],
      ["T3 Chikari", tonicHz * 8.0],
    ];
    refGrid.innerHTML = rows.map(([label, hz]) => `
      <div class="metric">
        <div class="metric-label">${label}</div>
        <div class="metric-value">${hz.toFixed(2)} Hz</div>
      </div>`).join("");
  }

  function renderMonitor(vres, tonicHz) {
    if (!vres.available) {
      metricsCard.classList.add("hidden");
      depthCard.classList.add("hidden");
      bannerSet(masterBanner, "banner-warning", `⏳ ${vres.error || "Diagnostic engine buffering audio…"}`);
      rawJsonEl.textContent = JSON.stringify(vres, null, 2);
      return;
    }

    const tuning = vres.tuning || {};
    const quality = vres.quality || {};
    const isHealthy = !!vres.is_healthy;
    const isVeena = vres.is_veena !== false;
    const soundType = vres.sound_type || "Veena String Resonance";
    const masterStatus = vres.status || "Unknown";

    const tstat = tuning.status || "NO_PITCH";
    const cdev = tuning.cents_dev || 0.0;
    const f0 = tuning.f0_hz || 0.0;
    const tgt = tuning.target_hz || 0.0;
    const sname = tuning.string_name || "—";
    const tmsg = tuning.message || "";

    const qok = !!quality.is_healthy;
    const qlabel = quality.label || "Healthy";
    const qconf = quality.confidence || 0.0;
    const qcls = quality.fault_class !== undefined ? quality.fault_class : 0;

    // 1. Master verdict banner
    if (masterStatus === "Silence" || tstat === "SILENCE" || qcls === -2) {
      bannerSet(masterBanner, "banner-info", "⚪ <strong>SYSTEM IDLE / SILENCE</strong> — Waiting for audio input. Pluck a Saraswati Veena string to begin diagnostics.");
    } else if (!isVeena || masterStatus === "Non-Veena Sound Detected" || tstat === "NON_VEENA" || qcls === -1) {
      bannerSet(masterBanner, "banner-error", `🚨 <strong>ANOMALY DETECTED — ${qlabel.toUpperCase()}</strong> (Acoustic signature does not match Saraswati Veena string).`);
    } else if (isHealthy && tstat === "IN_TUNE") {
      bannerSet(masterBanner, "banner-success", "🟢 <strong>HEALTHY &amp; IN TUNE</strong> — Instrument is structurally sound and tuned accurately within ±15 cents.");
    } else if (isHealthy && (tstat === "FLAT" || tstat === "SHARP")) {
      bannerSet(masterBanner, "banner-warning", `🟡 <strong>HEALTHY (TUNING WATCH: ${tstat})</strong> — Instrument structure is sound, but string is ${tstat} by ${Math.abs(cdev).toFixed(1)} cents.`);
    } else if (!isHealthy && qcls > 1) {
      bannerSet(masterBanner, "banner-error", `🚨 <strong>ANOMALY DETECTED — ${qlabel.toUpperCase()}</strong> (Structural defect identified by ML Classifier).`);
    } else if (!isHealthy && (tstat === "FLAT" || tstat === "SHARP")) {
      bannerSet(masterBanner, "banner-warning", `🟡 <strong>TUNING MISALIGNMENT: ${tstat}</strong> — String pitch is ${tstat} by ${Math.abs(cdev).toFixed(1)} cents.`);
    } else if (!isHealthy && tstat === "NO_PITCH") {
      bannerSet(masterBanner, "banner-warning", "⏳ <strong>ACOUSTIC DAMPENING / NO PITCH</strong> — Unclear fundamental string pitch detected.");
    } else {
      bannerSet(masterBanner, "banner-info", "ℹ️ <strong>DIAGNOSTIC ACTIVE</strong> — Analyzing resonance and tuning...");
    }

    // 2. Key metrics
    metricsCard.classList.remove("hidden");

    let healthBadge, healthDelta, healthCls;
    if (masterStatus === "Silence" || qcls === -2) {
      healthBadge = "🔇 Silence / Idle"; healthDelta = "No Input"; healthCls = "off";
    } else if (!isVeena || qcls === -1) {
      healthBadge = `🚨 ${qlabel}`; healthDelta = "Non-Veena Anomaly"; healthCls = "down";
    } else {
      healthBadge = isHealthy ? "✅ Healthy" : `🚨 ${qlabel}`;
      healthDelta = qconf > 0 ? `${qconf.toFixed(1)}% confidence` : "Active";
      healthCls = isHealthy ? "up" : "down";
    }
    document.getElementById("metric-health-value").textContent = healthBadge;
    setDelta("metric-health-delta", healthDelta, healthCls);

    let tLabel, tDelta, tCls;
    if (!isVeena || tstat === "NON_VEENA") {
      tLabel = "🚫 Non-Veena"; tDelta = "Non-instrument sound"; tCls = "down";
    } else if (tstat === "SILENCE") {
      tLabel = "🔇 Silence"; tDelta = "No pitch"; tCls = "off";
    } else {
      tLabel = `${STATUS_EMOJI[tstat] || "?"} ${tstat}`;
      tDelta = f0 > 0 ? `${cdev >= 0 ? "+" : ""}${cdev.toFixed(1)} cents` : "No pitch";
      tCls = tstat === "IN_TUNE" ? "up" : "down";
    }
    document.getElementById("metric-tuning-label").textContent = `Tuning (${sname})`;
    document.getElementById("metric-tuning-value").textContent = tLabel;
    setDelta("metric-tuning-delta", tDelta, tCls);

    const depthEntry = STRUCTURAL_DEPTH_DESCRIPTIONS[String(qcls)] || { title: qlabel, summary: "" };
    document.getElementById("metric-anomaly-value").textContent = depthEntry.title;
    setDelta("metric-anomaly-delta", qcls >= 0 ? `Class ${qcls}` : "Sound Validation",
      isHealthy ? "up" : (qcls === -2 ? "off" : "down"));

    // Tuning gauge
    const progressFill = document.getElementById("tuning-progress-fill");
    const progressText = document.getElementById("tuning-progress-text");
    if (isVeena && f0 > 0 && tgt > 0) {
      const clamped = Math.max(0, Math.min(1, (cdev + 50.0) / 100.0));
      progressFill.style.width = (clamped * 100).toFixed(1) + "%";
      progressText.textContent = `Detected: ${f0.toFixed(1)} Hz │ Target: ${tgt.toFixed(1)} Hz │ Dev: ${cdev >= 0 ? "+" : ""}${cdev.toFixed(1)} cents (Target: ±15 cents)`;
    } else if (!isVeena) {
      progressFill.style.width = "0%";
      progressText.textContent = `Non-Veena Acoustic Event (${soundType}) │ Requires genuine Veena string excitation`;
    } else {
      progressFill.style.width = "0%";
      progressText.textContent = "";
    }
    document.getElementById("tuning-guide-caption").textContent = tmsg ? `📌 Tuning Guide: ${tmsg}` : "";

    // 3. Structural depth
    depthCard.classList.remove("hidden");
    document.getElementById("structural-depth-title").textContent = depthEntry.title;
    document.getElementById("structural-depth-summary").textContent =
      depthEntry.summary || "Structural analysis performed via YAMNet embeddings.";
    document.getElementById("veena-session-caption").textContent =
      `Evaluated Target Session: 📄 ${vres.prefix || "—"} │ Sound Type: ${soundType} │ Feature Vector: 527-D Parallel Hybrid`;

    rawJsonEl.textContent = JSON.stringify(vres, null, 2);
  }

  function setDelta(id, text, cls) {
    const el = document.getElementById(id);
    el.textContent = text;
    el.className = "metric-delta " + (cls || "");
  }

  let inFlight = false;
  async function poll() {
    if (inFlight) return;
    inFlight = true;
    try {
      const tonicHz = TONIC_OPTIONS[tonicSelect.value];
      const stringVal = STRING_OPTIONS[stringSelect.value];
      renderReferenceTable(tonicHz);

      const params = new URLSearchParams({ tonic_hz: tonicHz, cents_threshold: "15" });
      if (stringVal !== null && stringVal !== undefined) params.set("string_label", stringVal);

      const res = await fetch("/api/veena_analysis?" + params.toString());
      if (res.ok) {
        const data = await res.json();
        renderMonitor(data, tonicHz);
      }
    } catch (e) {
      // network hiccup — keep last rendered state, try again next tick
    } finally {
      inFlight = false;
    }
  }

  tonicSelect.addEventListener("change", poll);
  stringSelect.addEventListener("change", poll);

  renderReferenceTable(TONIC_OPTIONS[tonicSelect.value]);
  poll();
  setInterval(poll, 2500); // matches the old AI_REFRESH_SEC
})();
