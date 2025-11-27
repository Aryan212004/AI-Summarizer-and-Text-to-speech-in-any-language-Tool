import axios from 'axios';

export const summarizeText = async (text) => {
  try {
    const res = await axios.post('http://127.0.0.1:5050/summarize', {
      text: text
    });
    return res.data.summary;
  } catch (err) {
    console.error("Error summarizing:", err);
    return "⚠️ Failed to summarize text.";
  }
};
