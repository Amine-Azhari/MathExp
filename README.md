Heyy, I developed a web app that can recognize a hand written math expression and evaluates it, and shows the result!

https://github.com/user-attachments/assets/04bdfdb2-b2a1-4f9e-a3ad-a6e86f1361a5

You can try it using this link : https://mathexp.onrender.com/ 


Curious to know how it works? Read below.
========== WORKFLOW ==========
• First, I trained a CNN (Convolutional Neural Network) model on digits (0–9) and basic operators (+, −, ×, /). For digits, I used the MNIST dataset, and for operators, I used a free dataset I found on Kaggle.
• Then, I built a small web app using Flask, HTML, CSS, and JavaScript. It allows users to draw a digit or an operator, and the app predicts what it is.
• After that, I needed to detect a sequence of elements (digits/operators), not just a single one. So I captured the canvas image, preprocessed it (blur, resize, etc. for better results), and applied OpenCV’s ‘connectedComponentsWithStats’ function to detect and isolate each connected component (each digit/operator) in the image.
• Once each element is extracted from the image, the model predicts what it is and stores it in an array sorted from left to right. Example: ["1", "1", "+", "7"]
• Finally, I built a small calculator that takes this array and evaluates it. The result is: 18.

