import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import axios from "axios";

const app = express();
app.use(cors());
app.use(bodyParser.json());

// Summarize Endpoint
app.post("/summarize", async (req, res) => {
  const { text } = req.body;
  if (!text || text.trim() === "") {
    return res.status(400).json({ error: "Text is required for summarization." });
  }

  try {
    // Example: using OpenAI API (replace with your own key)
    const response = await axios.post(
      "https://api.openai.com/v1/chat/completions",
      {
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: "You are a text summarizer." },
          { role: "user", content: `Summarize this: ${text}` }
        ],
        max_tokens: 200
      },
      {
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`
        }
      }
    );

    const summary = response.data.choices[0].message.content.trim();
    res.json({ summary });

  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Failed to summarize text." });
  }
});

// Translation Endpoint
app.post("/translate", async (req, res) => {
  const { text, targetLang } = req.body;
  if (!text || !targetLang) {
    return res.status(400).json({ error: "Text and targetLang are required." });
  }

  try {
    const translateRes = await axios.get("https://translate.googleapis.com/translate_a/single", {
      params: {
        client: "gtx",
        sl: "en",
        tl: targetLang,
        dt: "t",
        q: text
      }
    });

    const translation = translateRes.data[0].map(item => item[0]).join("");
    res.json({ translation });

  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: "Failed to translate text." });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`✅ MCP Server running on port ${PORT}`));
