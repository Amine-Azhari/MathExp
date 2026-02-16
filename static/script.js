const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const exprSpan = document.getElementById("expr");
const resultSpan = document.getElementById("result");

function initCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "white";
  ctx.lineWidth = 16;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
}

initCanvas();

let drawing = false;
let lastX = 0;
let lastY = 0;

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top,
  };
}

// Pointer Events (mouse + touch + pen)
canvas.addEventListener("pointerdown", (e) => {
  // Prevent scrolling/zooming while drawing
  e.preventDefault();

  drawing = true;
  const p = getPos(e);
  lastX = p.x;
  lastY = p.y;

  // Capture pointer so drawing continues even if finger leaves canvas
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  e.preventDefault();

  const p = getPos(e);

  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();

  lastX = p.x;
  lastY = p.y;
});

function stopDrawing(e) {
  if (!drawing) return;
  e.preventDefault();
  drawing = false;
  try {
    canvas.releasePointerCapture(e.pointerId);
  } catch (_) {
    // ignore
  }
}

canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);

// Buttons
document.getElementById("clearBtn").addEventListener("click", () => {
  initCanvas();
  exprSpan.textContent = "—";
  resultSpan.textContent = "—";
});

document.getElementById("predictBtn").addEventListener("click", async () => {
  const dataURL = canvas.toDataURL("image/png");

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataURL }),
  });

  const data = await res.json();
  if (data.error) {
    exprSpan.textContent = "Error";
    console.error(data.error);
    return;
  }

  exprSpan.textContent = data.expr || "—";
  resultSpan.textContent = data.result ?? "_";
});
