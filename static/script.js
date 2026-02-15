const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const exprSpan = document.getElementById("expr");
const resultSpan = document.getElementById("result")


function initCanvas() {
  // black background
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // drawing style
  ctx.strokeStyle = "white";
  ctx.lineWidth = 16;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
}

initCanvas();

let drawing = false;
let lastX = 0;
let lastY = 0;


canvas.addEventListener("mousedown", (e) => {
  drawing = true;

  const rect = canvas.getBoundingClientRect();
  lastX = e.clientX - rect.left;
  lastY = e.clientY - rect.top;
});
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseleave", () => drawing = false);

canvas.addEventListener("mousemove", (e) => {
  if (!drawing) return;

  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();

  lastX = x;
  lastY = y;
});


document.getElementById("clearBtn").addEventListener("click", () => {
  initCanvas();
  exprSpan.textContent = "—";
});

document.getElementById("predictBtn").addEventListener("click", async () => {
  const dataURL = canvas.toDataURL("image/png");

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: dataURL })
  });

  const data = await res.json();
  if (data.error) {
    exprSpan.textContent = "Error";
    console.error(data.error);
    return;
  }

  // show the detected expression
  // data.expr could be "2+3"
  exprSpan.textContent = data.expr || "—";
  resultSpan.textContent = data.result || "_";
});
