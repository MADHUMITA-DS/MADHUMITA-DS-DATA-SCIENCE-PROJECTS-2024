// Selecting page elements
const instructionsPage = document.getElementById("instructions-page");
const webcamPage = document.getElementById("webcam-page");
const startButton = document.getElementById("start-button");
const backButton = document.getElementById("back-button");
const videoFeed = document.getElementById("video-feed");
const predictionDisplay = document.getElementById("prediction-display");

function startApplication() {
    // Run the batch file when the button is clicked
    window.open("start_application.bat");
}

// Show Webcam Page and hide Instructions Page
startButton.addEventListener("click", () => {
    instructionsPage.style.display = "none";
    webcamPage.style.display = "flex";
    startWebcam();
});

// Show Instructions Page and hide Webcam Page
backButton.addEventListener("click", () => {
    stopWebcam();
    webcamPage.style.display = "none";
    instructionsPage.style.display = "flex";
});

// Start webcam feed
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoFeed.srcObject = stream;
        // Start the real-time prediction function
        updatePrediction();
    } catch (err) {
        alert("Webcam access denied or not supported by your browser.");
    }
}

// Stop webcam feed
function stopWebcam() {
    let stream = videoFeed.srcObject;
    let tracks = stream.getTracks();
    tracks.forEach(track => track.stop());
    videoFeed.srcObject = null;
}

// Simulate updating predictions
function updatePrediction() {
    // This is a placeholder function
    const predictionTexts = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z"
    ];
    let index = 0;
    setInterval(() => {
        predictionDisplay.value = `Predicted Character: ${predictionTexts[index]}`;
        index = (index + 1) % predictionTexts.length;
    }, 1000); // Update every second
}
