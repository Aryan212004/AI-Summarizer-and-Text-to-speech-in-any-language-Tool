import axios from 'axios';

export async function summarizeText(text) {
  try {
    const response = await axios.post('http://localhost:5050/summarize', {
      text,
    });
    console.log("✅ Response:", response.data);
    return response.data.summary;
  } catch (err) {
    console.error("❌ Error calling summarizer:", err.message);
    return "Summarization failed.";
  }
}
