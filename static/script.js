const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultSpan = document.getElementById("result");

function initCanvas() {
  // black background
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // drawing style
  ctx.strokeStyle = "white";
  ctx.lineWidth = 16;
  ctx.lineCap = "round";
}

initCanvas();

let drawing = false;

canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseleave", () => drawing = false);

canvas.addEventListener("mousemove", (e) => {
  if (!drawing) return;

  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x, y);
  ctx.stroke();
});

document.getElementById("clearBtn").addEventListener("click", () => {
  initCanvas();
  resultSpan.textContent = "—";
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
    resultSpan.textContent = "Error";
    console.error(data.error);
  } else {
    switch (data.prediction){
        case 10:
            resultSpan.textContent = "+"; break;
        case 11:
            resultSpan.textContent = "-"; break;
        case 12:
            resultSpan.textContent = "*"; break;
        case 13:
            resultSpan.textContent = "/"; break;
        default :
            resultSpan.textContent = data.prediction; break;
    }
  }
});
