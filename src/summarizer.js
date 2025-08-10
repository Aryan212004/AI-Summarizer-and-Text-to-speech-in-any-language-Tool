// summarizer.js
import axios from 'axios';

export const summarizeText = async (text) => {
  try {
    const res = await axios.post('http://localhost:5050/summarize', { text });
    return res.data.summary;
  } catch (err) {
    console.error('Error summarizing:', err);
    return '⚠️ Failed to connect to summarizer API.';
  }
};
