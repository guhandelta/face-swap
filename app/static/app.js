const state = {
  source: { file: null, faces: [], selected: 0 },
  target: { file: null, faces: [], selected: -1 }, // -1 = every face
};
const $ = (s, el = document) => el.querySelector(s);
const errorEl = $("#error");
const goBtn = $("#go");
const consent = $("#consent");

async function checkHealth() {
  try {
    const h = await (await fetch("/health")).json();
    $("#status").textContent = h.models_loaded
      ? `Ready · ${h.providers[0].replace("ExecutionProvider", "")}`
      : "Models are not loaded. Run scripts/download_models.py and restart.";
    for (const opt of document.querySelectorAll("#restorer option")) {
      if (!["auto", "none"].includes(opt.value) && !h.restorers.includes(opt.value)) {
        opt.disabled = true; opt.textContent += " (not installed)";
      }
    }
    if (!h.masks.face_parser && !h.masks.occluder) {
      $("#occlusion").checked = false; $("#occlusion").disabled = true;
      $("#occlusion").parentElement.title = "Face parsing models are not installed";
    }
  } catch {
    $("#status").textContent = "Server unreachable.";
  }
}

function refreshButton() {
  goBtn.disabled = !(state.source.faces.length && state.target.faces.length && consent.checked);
}

function drawBoxes(role) {
  const picker = $(`.picker[data-role="${role}"]`);
  const img = $("img", picker);
  const layer = $(".boxes", picker);
  const s = state[role];
  layer.innerHTML = "";
  if (!img.naturalWidth) return;
  const k = img.clientWidth / img.naturalWidth;
  const offX = img.offsetLeft, offY = img.offsetTop;
  s.faces.forEach((f) => {
    const [x1, y1, x2, y2] = f.bbox.map((v) => v * k * (s.scale || 1));
    const box = document.createElement("div");
    box.className = "box" + (s.selected === -1 || s.selected === f.index ? " sel" : "");
    Object.assign(box.style, { left: `${offX + x1}px`, top: `${offY + y1}px`, width: `${x2 - x1}px`, height: `${y2 - y1}px` });
    box.innerHTML = `<span>${f.index + 1}</span>`;
    box.addEventListener("click", (e) => {
      e.preventDefault(); e.stopPropagation();
      if (role === "target") s.selected = s.selected === f.index ? -1 : f.index;
      else s.selected = f.index;
      drawBoxes(role); note(role);
    });
    layer.appendChild(box);
  });
}

function note(role) {
  const s = state[role];
  const el = $(`.picker[data-role="${role}"] .faces-note`);
  if (!s.file) return (el.textContent = "");
  if (!s.faces.length) return (el.textContent = "No face found in this image.");
  const n = s.faces.length;
  if (role === "source") el.textContent = n > 1 ? `${n} faces. Using face ${s.selected + 1}; click a box to change.` : "1 face found.";
  else el.textContent = s.selected === -1
    ? `Replacing ${n > 1 ? `all ${n} faces` : "the face"}.${n > 1 ? " Click one to replace only it." : ""}`
    : `Replacing face ${s.selected + 1} only. Click it again to replace all.`;
}

async function pick(role, file) {
  if (!file || !file.type.startsWith("image/")) return;
  const picker = $(`.picker[data-role="${role}"]`);
  const img = $("img", picker);
  const s = state[role];
  Object.assign(s, { file, faces: [], selected: role === "source" ? 0 : -1 });
  errorEl.textContent = "";
  img.src = URL.createObjectURL(file);
  $(".stage", picker).classList.add("has-img");
  $(".faces-note", picker).textContent = "Finding faces…";
  refreshButton();

  const body = new FormData();
  body.append("image", file);
  try {
    const r = await fetch("/api/detect", { method: "POST", body });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    await img.decode().catch(() => {});
    s.faces = data.faces;
    s.scale = img.naturalWidth / data.width; // server may downscale large images
  } catch (e) {
    errorEl.textContent = e.message;
  }
  drawBoxes(role); note(role); refreshButton();
}

document.querySelectorAll(".picker").forEach((picker) => {
  const role = picker.dataset.role;
  const input = $("input", picker);
  const stage = $(".stage", picker);
  input.addEventListener("change", () => pick(role, input.files[0]));
  stage.addEventListener("dragover", (e) => { e.preventDefault(); stage.classList.add("over"); });
  stage.addEventListener("dragleave", () => stage.classList.remove("over"));
  stage.addEventListener("drop", (e) => {
    e.preventDefault(); stage.classList.remove("over");
    pick(role, e.dataTransfer.files[0]);
  });
});
window.addEventListener("resize", () => { drawBoxes("source"); drawBoxes("target"); });
consent.addEventListener("change", refreshButton);

goBtn.addEventListener("click", async () => {
  errorEl.textContent = "";
  goBtn.disabled = true;
  goBtn.textContent = "Swapping…";
  const body = new FormData();
  body.append("source", state.source.file);
  body.append("target", state.target.file);
  body.append("source_face_index", state.source.selected);
  body.append("target_face_index", state.target.selected);
  body.append("swap_resolution", $("#res").value);
  body.append("restorer", $("#restorer").value);
  body.append("blend_mode", $("#blend").value);
  body.append("color_match", $("#color").checked);
  body.append("lighting", $("#lighting").checked ? 0.8 : 0);
  body.append("occlusion", $("#occlusion").checked);
  body.append("sharpness_match", $("#sharp").checked);
  body.append("grain_match", $("#grain").checked);
  body.append("consent", consent.checked);
  try {
    const r = await fetch("/api/swap", { method: "POST", body });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const warns = JSON.parse(r.headers.get("X-FastSwap-Warnings") || "[]");
    $("#warnings").innerHTML = "";
    for (const w of warns) {
      const li = document.createElement("li"); li.textContent = w; $("#warnings").appendChild(li);
    }
    const url = URL.createObjectURL(await r.blob());
    $("#result").src = url;
    $("#download").href = url;
    $(".result").hidden = false;
    $(".result").scrollIntoView({ behavior: "smooth" });
  } catch (e) {
    errorEl.textContent = e.message;
  } finally {
    goBtn.textContent = "Swap faces";
    refreshButton();
  }
});

checkHealth();
