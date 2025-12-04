const axios = require("axios");

async function uploadSong() {
    const fileInput = document.getElementById("audioFile");
    const resultBox = document.getElementById("resultBox");

    if (fileInput.files.length === 0) {
        alert("Choose an audio file!");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultBox.textContent = "Processing... Please wait...";

    try {
        const res = await axios.post(
            "http://127.0.0.1:8000/analyze",
            formData,
            { headers: { "Content-Type": "multipart/form-data" } }
        );

        resultBox.textContent = JSON.stringify(res.data, null, 2);
    }
    catch (err) {
        resultBox.textContent = "Error: " + err;
    }
}
