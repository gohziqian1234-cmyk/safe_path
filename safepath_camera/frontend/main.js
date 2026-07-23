const video = document.getElementById("camera");
const canvas = document.getElementById("frame");

const state = {
  initialized: false,
  stream: null,
  timer: null,
  waitingForServer: false,
  requestedFacingMode: "environment",
  activeFacingMode: "",
  args: null,
};

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
  sendStatus("starting");

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
        sendStatus("error", {
          error:
            genericError?.message ||
            fallbackError?.message ||
            preferredError?.message ||
            "Camera access failed.",
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

  canvas.width = state.args.width;
  canvas.height = state.args.height;
  const context = canvas.getContext("2d", { alpha: false });
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  // JPEG is substantially smaller than the PNG used by camera_input_live.
  const image = canvas.toDataURL("image/jpeg", state.args.jpegQuality);
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
  Streamlit.setFrameHeight(0);

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
  state.waitingForServer = false;
  if (facingChanged) {
    startCamera(nextArgs);
  } else {
    scheduleCapture();
  }
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
Streamlit.setFrameHeight(0);
window.addEventListener("beforeunload", stopStream);
