import { useState, useEffect } from 'react';
import { summarizeText } from './summarizer';
import axios from 'axios';

function App() {
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState('en');
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [translatedSummary, setTranslatedSummary] = useState('');
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('theme') === 'dark');

  const translateText = async (text, lang, setOutput) => {
    if (lang === 'en') {
      setOutput(text);
      return;
    }

    try {
      const res = await axios.get('https://translate.googleapis.com/translate_a/single', {
        params: {
          client: 'gtx',
          sl: 'en',
          tl: lang,
          dt: 't',
          q: text,
        },
      });
      const translated = res.data[0].map((item) => item[0]).join('');
      setOutput(translated);
    } catch (err) {
      console.error('Translation failed:', err.message);
      setOutput('⚠️ Translation failed.');
    }
  };

  useEffect(() => {
    stopSpeech();
    translateText(inputText, language, setTranslatedText);
    if (summary) {
      translateText(summary, language, setTranslatedSummary);
    }
  }, [language, summary, inputText]);

  const handleSummarize = async () => {
    stopSpeech();
    if (!inputText.trim()) {
      alert("⚠️ Please paste or type some text first.");
      return;
    }
    setLoading(true);
    const result = await summarizeText(inputText);
    setSummary(result);
    translateText(result, language, setTranslatedSummary);
    setLoading(false);
  };

  const speak = (text, lang = 'en') => {
    if (!text || text.startsWith('⚠️')) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang =
      lang === 'hi' ? 'hi-IN' :
      lang === 'fr' ? 'fr-FR' :
      lang === 'de' ? 'de-DE' :
      lang === 'es' ? 'es-ES' :
      'en-US';
    speechSynthesis.cancel();
    speechSynthesis.speak(utterance);
  };

  const stopSpeech = () => {
    speechSynthesis.cancel();
  };

  const toggleTheme = () => {
    const newTheme = !darkMode;
    setDarkMode(newTheme);
    document.documentElement.classList.toggle('dark', newTheme);
    localStorage.setItem('theme', newTheme ? 'dark' : 'light');
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      alert('📋 Copied to clipboard!');
    });
  };

  return (
    <div className="min-h-screen bg-white dark:bg-gray-900 text-black dark:text-white p-6 transition-colors duration-500">
      <header className="text-4xl font-bold mb-6 text-center">🧠 AI Text Summarizer</header>

      <div className="flex flex-col md:flex-row justify-between items-center mb-4 gap-4">
        <div className="flex flex-col md:flex-row gap-2 items-center">
          <button
            onClick={handleSummarize}
            className="bg-blue-600 px-4 py-2 rounded text-white hover:bg-blue-700 transition"
          >
            {loading ? 'Summarizing...' : 'Summarize'}
          </button>

          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="text-black dark:text-white bg-gray-200 dark:bg-gray-800 px-2 py-1 rounded"
          >
            <option value="en">English</option>
            <option value="hi">Hindi</option>
            <option value="fr">French</option>
            <option value="es">Spanish</option>
            <option value="de">German</option>
          </select>
        </div>

        <button
          onClick={toggleTheme}
          className="bg-yellow-500 text-black px-3 py-2 rounded hover:bg-yellow-600 transition"
        >
          {darkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
        </button>
      </div>

      {/* Input Bar */}
      <textarea
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        placeholder="Paste your article or text here..."
        className="w-full h-40 p-4 border rounded-lg bg-gray-100 dark:bg-gray-800 dark:text-white resize-none mb-4"
      />

      {/* Article Section */}
      {inputText && (
        <div className="bg-gray-100 dark:bg-gray-800 p-6 rounded-2xl shadow-md transition-colors duration-500">
          <div className="flex justify-between items-center mb-2">
            <h2 className="text-xl font-semibold">Original Text</h2>
            <div className="flex gap-2">
              <button
                onClick={() => speak(translatedText, language)}
                className="bg-green-600 px-3 py-1 rounded text-white hover:bg-green-700"
              >
                🔊 Read
              </button>
              <button
                onClick={() => copyToClipboard(translatedText)}
                className="bg-gray-600 px-3 py-1 rounded text-white hover:bg-gray-700"
              >
                📋 Copy
              </button>
              <button
                onClick={stopSpeech}
                className="bg-red-600 px-3 py-1 rounded text-white hover:bg-red-700"
              >
                ⏹ Stop
              </button>
            </div>
          </div>
          {translatedText.split('\n').map((para, i) => (
            <p key={i} className="mb-4">{para}</p>
          ))}
        </div>
      )}

      {/* Summary Section */}
      {summary && (
        <div className="mt-6 p-4 bg-gray-200 dark:bg-gray-700 rounded-lg">
          <div className="flex justify-between items-center mb-2">
            <h3 className="font-semibold text-lg">📝 Summary:</h3>
            <div className="flex gap-2">
              <button
                onClick={() => speak(translatedSummary, language)}
                className="bg-purple-600 px-3 py-1 rounded text-white hover:bg-purple-700"
              >
                🔊 Read Summary
              </button>
              <button
                onClick={() => copyToClipboard(translatedSummary)}
                className="bg-gray-600 px-3 py-1 rounded text-white hover:bg-gray-700"
              >
                📋 Copy
              </button>
              <button
                onClick={stopSpeech}
                className="bg-red-600 px-3 py-1 rounded text-white hover:bg-red-700"
              >
                ⏹ Stop
              </button>
            </div>
          </div>
          <p>{translatedSummary}</p>
        </div>
      )}
    </div>
  );
}

export default App;
