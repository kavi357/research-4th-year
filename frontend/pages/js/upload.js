async function analyze() {
  const fileInput = document.getElementById("audioFile");
  const status = document.getElementById("status");
  const resultsBox = document.getElementById("results");

  if (!fileInput.files.length) {
    status.innerText = "Please select a file";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  status.innerText = "Analyzing...";

  try {
    const res = await window.api.analyzeSong(formData);

    let text = `Query Song\n`;
    text += `Tempo: ${res.query.tempo}\n`;
    text += `Pitch Median: ${res.query.pitch_median}\n\n`;

    res.top_matches.forEach((r, i) => {
      text += `#${i+1} Track ${r.track_id}\n`;
      text += `Tempo Similarity: ${r.tempo_similarity}%\n`;
      text += `Pitch Similarity: ${r.pitch_similarity}%\n`;
      text += `Overall Score: ${r.overall_score}%\n\n`;
    });

    resultsBox.innerText = text;
    status.innerText = "Done";

  } catch (e) {
    status.innerText = "Error occurred";
    console.error(e);
  }
}
