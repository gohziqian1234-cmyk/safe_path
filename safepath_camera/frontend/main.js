const video = document.getElementById("camera");
const captureCanvas = document.getElementById("capture-frame");
const overlayCanvas = document.getElementById("overlay");
const previewShell = document.getElementById("preview-shell");
const cameraStatus = document.getElementById("camera-status");

const state = {
  initialized: false,
  stream: null,
  timer: null,
  waitingForServer: false,
  requestedFacingMode: "environment",
  activeFacingMode: "",
  args: null,
  lastOverlayRevision: null,
};

function setCameraStatus(message, status = "starting") {
  cameraStatus.textContent = message;
  cameraStatus.dataset.state = status;
}

function updateFrameHeight() {
  Streamlit.setFrameHeight(Math.ceil(previewShell.getBoundingClientRect().height));
}

function sendStatus(status, extra = {}) {
  Streamlit.setComponentValue({
    status,
    image: null,
    requestedFacingMode: state.requestedFacingMode,
    activeFacingMode: state.activeFacingMode,
    error: "",
    capturedAtEpochMs: null,
    ...extra,
  });
}

function stopStream() {
  if (state.timer !== null) {
    window.clearTimeout(state.timer);
    state.timer = null;
  }
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
}

function drawOverlay(overlay) {
  const width = state.args?.width || 640;
  const height = state.args?.height || 480;
  if (overlayCanvas.width !== width) {
    overlayCanvas.width = width;
  }
  if (overlayCanvas.height !== height) {
    overlayCanvas.height = height;
  }

  const context = overlayCanvas.getContext("2d");
  context.clearRect(0, 0, width, height);
  if (!overlay || !Array.isArray(overlay.zonePoints)) {
    return;
  }

  const highRisk = overlay.riskLevel === "HIGH";
  const zoneColor = highRisk ? "#ef4444" : "#22c55e";
  const zoneFill = highRisk
    ? "rgba(239, 68, 68, 0.18)"
    : "rgba(34, 197, 94, 0.16)";

  context.beginPath();
  overlay.zonePoints.forEach(([x, y], index) => {
    if (index === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  });
  context.closePath();
  context.fillStyle = zoneFill;
  context.fill();
  context.strokeStyle = zoneColor;
  context.lineWidth = 3;
  context.stroke();

  context.font = "600 16px Inter, sans-serif";
  context.textBaseline = "bottom";
  for (const detection of overlay.detections || []) {
    const [x1, y1, x2, y2] = detection.box;
    const color =
      detection.label.toLowerCase() === "person"
        ? "#f59e0b"
        : detection.inDangerZone
          ? "#ef4444"
          : "#22c55e";
    context.strokeStyle = color;
    context.lineWidth = 3;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const confidence = Math.round(detection.confidence * 100);
    const label = `${detection.label} ${confidence}%`;
    const labelWidth = context.measureText(label).width + 12;
    const labelY = Math.max(22, y1);
    context.fillStyle = color;
    context.fillRect(x1, labelY - 22, labelWidth, 22);
    context.fillStyle = "#ffffff";
    context.fillText(label, x1 + 6, labelY - 3);
  }

  context.fillStyle = highRisk
    ? "rgba(220, 38, 38, 0.92)"
    : "rgba(22, 101, 52, 0.88)";
  context.fillRect(0, 0, width, 38);
  context.fillStyle = "#ffffff";
  context.font = "700 18px Inter, sans-serif";
  context.textBaseline = "middle";
  context.fillText(`SAFEPATH RISK: ${overlay.riskLevel || "LOW"}`, 14, 19);
}

function cameraConstraints(facingMode, args) {
  return {
    audio: false,
    video: {
      facingMode: { exact: facingMode },
      width: { ideal: args.width },
      height: { ideal: args.height },
      frameRate: { ideal: 15, max: 20 },
    },
  };
}

async function requestCamera(facingMode, args) {
  return navigator.mediaDevices.getUserMedia(
    cameraConstraints(facingMode, args),
  );
}

async function startCamera(args) {
  stopStream();
  state.requestedFacingMode = args.facingMode;
  state.activeFacingMode = "";
  state.waitingForServer = true;
  setCameraStatus("Starting camera…");

  const fallbackFacing =
    args.facingMode === "environment" ? "user" : "environment";

  try {
    state.stream = await requestCamera(args.facingMode, args);
  } catch (preferredError) {
    try {
      state.stream = await requestCamera(fallbackFacing, args);
    } catch (fallbackError) {
      try {
        state.stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            width: { ideal: args.width },
            height: { ideal: args.height },
          },
        });
      } catch (genericError) {
        const message =
          genericError?.message ||
          fallbackError?.message ||
          preferredError?.message ||
          "Camera access failed.";
        setCameraStatus(message, "error");
        sendStatus("error", {
          error: message,
        });
        return;
      }
    }
  }

  video.srcObject = state.stream;
  await video.play();
  const settings = state.stream.getVideoTracks()[0]?.getSettings?.() || {};
  state.activeFacingMode = settings.facingMode || fallbackFacing;
  state.waitingForServer = true;
  setCameraStatus("Camera live · starting AI", "ready");
  updateFrameHeight();
  sendStatus("ready");
}

function scheduleCapture() {
  if (
    state.timer !== null ||
    state.waitingForServer ||
    !state.stream ||
    !state.args
  ) {
    return;
  }

  state.timer = window.setTimeout(() => {
    state.timer = null;
    captureFrame();
  }, state.args.intervalMs);
}

function captureFrame() {
  if (!state.stream || state.waitingForServer || video.readyState < 2) {
    scheduleCapture();
    return;
  }

  captureCanvas.width = state.args.width;
  captureCanvas.height = state.args.height;
  const context = captureCanvas.getContext("2d", { alpha: false });
  context.drawImage(
    video,
    0,
    0,
    captureCanvas.width,
    captureCanvas.height,
  );

  // JPEG is substantially smaller than the PNG used by camera_input_live.
  const image = captureCanvas.toDataURL("image/jpeg", state.args.jpegQuality);
  const capturedAtEpochMs = Date.now();
  state.waitingForServer = true;
  Streamlit.setComponentValue({
    status: "ready",
    image,
    requestedFacingMode: state.requestedFacingMode,
    activeFacingMode: state.activeFacingMode,
    error: "",
    capturedAtEpochMs,
  });
}

function onRender(event) {
  const nextArgs = event.detail.args;
  previewShell.style.aspectRatio = `${nextArgs.width} / ${nextArgs.height}`;
  updateFrameHeight();

  if (!navigator.mediaDevices?.getUserMedia) {
    sendStatus("error", {
      error: "This browser does not support camera access.",
    });
    return;
  }

  if (!state.initialized) {
    state.initialized = true;
    state.args = nextArgs;
    startCamera(nextArgs);
    return;
  }

  const facingChanged = nextArgs.facingMode !== state.requestedFacingMode;
  state.args = nextArgs;
  if (nextArgs.overlayRevision !== state.lastOverlayRevision) {
    state.lastOverlayRevision = nextArgs.overlayRevision;
    drawOverlay(nextArgs.overlay);
    if (nextArgs.overlay?.processedFrames) {
      setCameraStatus(
        `Camera live · AI frame ${nextArgs.overlay.processedFrames}`,
        "ready",
      );
    }
  }
  state.waitingForServer = false;
  if (facingChanged) {
    startCamera(nextArgs);
  } else {
    scheduleCapture();
  }
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
updateFrameHeight();
video.addEventListener("loadedmetadata", updateFrameHeight);
window.addEventListener("resize", updateFrameHeight);
window.addEventListener("beforeunload", stopStream);
