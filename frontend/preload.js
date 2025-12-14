const { contextBridge } = require("electron");
const axios = require("axios");

contextBridge.exposeInMainWorld("api", {
  analyzeSong: async (formData) => {
    const res = await axios.post(
      "http://127.0.0.1:8000/analyze",
      formData,
      { headers: { "Content-Type": "multipart/form-data" } }
    );
    return res.data;
  }
});
