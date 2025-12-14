document.getElementById("uploadBtn").addEventListener("click", async () => {
    const fileInput = document.getElementById("audioFile");
    const status = document.getElementById("status");
    const stemsDiv = document.getElementById("stems");

    if (!fileInput.files.length) {
        status.innerText = "Select a song first";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]); // MUST match FastAPI field name

    status.innerText = "Uploading & processing...";

    try {
        const result = await window.api.uploadAudio(formData);
        console.log(result);

        if (result.status !== "success") {
            status.innerText = "Error: " + result.message;
            return;
        }

        status.innerText = "Separation complete!";

        stemsDiv.innerHTML = "";
        result.stems.forEach(stem => {
            const p = document.createElement("p");
            p.innerText = stem.name;

            const audio = document.createElement("audio");
            audio.controls = true;
            audio.src = "http://127.0.0.1:8000" + stem.url;

            stemsDiv.appendChild(p);
            stemsDiv.appendChild(audio);
        });

    } catch (err) {
        console.error(err);
        status.innerText = "Error: " + err;
    }
});
