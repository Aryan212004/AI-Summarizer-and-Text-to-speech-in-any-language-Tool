import { useState, useEffect } from 'react';

const API_BASE_URL = 'http://localhost:5050';

function App() {
  const [activeTab, setActiveTab] = useState('summarizer');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState('en');
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [translatedSummary, setTranslatedSummary] = useState('');
  const [darkMode, setDarkMode] = useState(false);
  const [error, setError] = useState('');
  
  // Sentiment Analysis States
  const [sentimentText, setSentimentText] = useState('');
  const [sentimentResult, setSentimentResult] = useState(null);
  const [analyzingsentiment, setAnalyzingsentiment] = useState(false);

  // Humanizer States
  const [humanizerText, setHumanizerText] = useState('');
  const [humanizedText, setHumanizedText] = useState('');
  const [humanizingText, setHumanizingText] = useState(false);

  const translateText = async (text, lang) => {
    if (lang === 'en' || !text) return text;
    
    try {
      const res = await fetch('https://translate.googleapis.com/translate_a/single?' + new URLSearchParams({
        client: 'gtx',
        sl: 'en',
        tl: lang,
        dt: 't',
        q: text,
      }));
      const data = await res.json();
      return data[0].map((item) => item[0]).join('');
    } catch (err) {
      console.error('Translation failed:', err.message);
      return '⚠️ Translation failed.';
    }
  };

  // Backend Summarizer
  const handleSummarize = async () => {
    stopSpeech();
    setError('');
    
    if (!inputText.trim()) {
      setError("⚠️ Please paste or type some text first.");
      return;
    }
    
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/summarize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: inputText }),
      });
      
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        setSummary('');
      } else {
        setSummary(data.summary);
        const translated = await translateText(data.summary, language);
        setTranslatedSummary(translated);
      }
    } catch (err) {
      setError('Failed to connect to backend. Make sure the server is running on port 5050.');
      console.error('Summarize error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Backend Humanizer
  const handleHumanize = async () => {
    stopSpeech();
    setError('');
    
    if (!humanizerText.trim()) {
      setError("⚠️ Please paste or type some text first.");
      return;
    }
    
    setHumanizingText(true);
    try {
      const response = await fetch(`${API_BASE_URL}/humanize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: humanizerText }),
      });
      
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        setHumanizedText('');
      } else {
        setHumanizedText(data.humanized);
      }
    } catch (err) {
      setError('Failed to connect to backend. Make sure the server is running on port 5050.');
      console.error('Humanize error:', err);
    } finally {
      setHumanizingText(false);
    }
  };

  // Backend Sentiment Analyzer
  const handleAnalyzeSentiment = async () => {
    if (!sentimentText.trim()) {
      setError('Please enter some text to analyze.');
      return;
    }
    
    setAnalyzingsentiment(true);
    setError('');
    
    try {
      const response = await fetch(`${API_BASE_URL}/sentiment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: sentimentText }),
      });
      
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        setSentimentResult(null);
      } else {
        // Highlight sentiment words
        const positiveWords = ['love', 'great', 'excellent', 'fantastic', 'amazing', 'wonderful', 'perfect', 'best', 'awesome', 'superb', 'outstanding', 'happy', 'satisfied', 'recommend', 'pleased', 'delighted', 'impressed', 'good', 'nice', 'easy', 'fast'];
        const negativeWords = ['hate', 'terrible', 'awful', 'poor', 'bad', 'horrible', 'disappointed', 'worst', 'regret', 'avoid', 'problem', 'issue', 'broken', 'frustrated', 'unhappy', 'dislike', 'angry', 'difficult', 'slow', 'expensive'];
        
        let highlighted = sentimentText;
        positiveWords.forEach(word => {
          const regex = new RegExp(`\\b${word}\\b`, 'gi');
          highlighted = highlighted.replace(regex, match => `<span class="bg-cyan-200 dark:bg-cyan-900 px-1 rounded">${match}</span>`);
        });
        
        negativeWords.forEach(word => {
          const regex = new RegExp(`\\b${word}\\b`, 'gi');
          highlighted = highlighted.replace(regex, match => `<span class="bg-purple-200 dark:bg-purple-900 px-1 rounded">${match}</span>`);
        });
        
        setSentimentResult({
          ...data,
          highlighted,
          sentimentClass: data.sentiment === 'Positive' ? 'positive' : data.sentiment === 'Negative' ? 'negative' : 'neutral',
          icon: data.sentiment === 'Positive' ? '😊' : data.sentiment === 'Negative' ? '😞' : '😐',
        });
      }
    } catch (err) {
      setError('Failed to connect to backend. Make sure the server is running on port 5050.');
      console.error('Sentiment error:', err);
    } finally {
      setAnalyzingsentiment(false);
    }
  };

  useEffect(() => {
    stopSpeech();
    if (inputText) {
      translateText(inputText, language).then(setTranslatedText);
    }
    if (summary) {
      translateText(summary, language).then(setTranslatedSummary);
    }
  }, [language, summary, inputText]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

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
    setDarkMode(!darkMode);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      alert('📋 Copied to clipboard!');
    });
  };

  const exampleTexts = [
    { type: 'positive', text: 'This product is absolutely fantastic! It exceeded all my expectations and works perfectly. I would recommend it to anyone.' },
    { type: 'neutral', text: 'The service was okay, but nothing special. It met the basic requirements but didn\'t impress me. I might consider other options next time.' },
    { type: 'negative', text: 'I\'m extremely disappointed with this purchase. The quality is poor and it stopped working after just two days. I would not recommend this to anyone.' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800 text-black dark:text-white transition-colors duration-500">
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <h1 className="text-3xl font-bold flex items-center gap-2">
              AI Text Analysis Suite
            </h1>
            <button
              onClick={toggleTheme}
              className="bg-yellow-400 text-gray-900 px-4 py-2 rounded-lg hover:bg-yellow-300 transition font-semibold"
            >
              {darkMode ? '☀️ Light Mode' : '🌙 Dark Mode'}
            </button>
          </div>
        </div>
      </header>

      {/* Tab Navigation */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        {error && (
          <div className="bg-red-100 dark:bg-red-900 border-l-4 border-red-600 p-4 mb-6 rounded">
            <p className="text-red-700 dark:text-red-200">{error}</p>
          </div>
        )}

        <div className="flex gap-4 mb-6 bg-white dark:bg-gray-800 p-2 rounded-xl shadow-md flex-wrap">
          <button
            onClick={() => setActiveTab('summarizer')}
            className={`flex-1 py-3 px-6 rounded-lg font-semibold transition ${
              activeTab === 'summarizer'
                ? 'bg-blue-600 text-white shadow-lg'
                : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            📝 Text Summarizer
          </button>
          <button
            onClick={() => setActiveTab('sentiment')}
            className={`flex-1 py-3 px-6 rounded-lg font-semibold transition ${
              activeTab === 'sentiment'
                ? 'bg-purple-600 text-white shadow-lg'
                : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            😊 Sentiment Analyzer
          </button>
          <button
            onClick={() => setActiveTab('humanizer')}
            className={`flex-1 py-3 px-6 rounded-lg font-semibold transition ${
              activeTab === 'humanizer'
                ? 'bg-green-600 text-white shadow-lg'
                : 'bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600'
            }`}
          >
            👤 Humanizer
          </button>
        </div>

        {/* Summarizer Tab */}
        {activeTab === 'summarizer' && (
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
              <h2 className="text-2xl font-bold mb-4 text-blue-600 dark:text-blue-400">
                Text Summarizer
              </h2>
              
              <div className="flex flex-col md:flex-row gap-4 mb-4">
                <button
                  onClick={handleSummarize}
                  disabled={loading}
                  className="bg-blue-600 px-6 py-3 rounded-lg text-white hover:bg-blue-700 transition font-semibold disabled:opacity-50"
                >
                  {loading ? '⏳ Summarizing...' : '✨ Summarize'}
                </button>

                <select
                  value={language}
                  onChange={(e) => setLanguage(e.target.value)}
                  className="text-black dark:text-white bg-gray-200 dark:bg-gray-700 px-4 py-2 rounded-lg border-2 border-gray-300 dark:border-gray-600"
                >
                  <option value="en">🇬🇧 English</option>
                  <option value="hi">🇮🇳 Hindi</option>
                  <option value="fr">🇫🇷 French</option>
                  <option value="es">🇪🇸 Spanish</option>
                  <option value="de">🇩🇪 German</option>
                </select>
              </div>

              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Paste your article or text here..."
                className="w-full h-40 p-4 border-2 rounded-xl bg-gray-50 dark:bg-gray-700 dark:text-white border-gray-300 dark:border-gray-600 resize-none mb-4 focus:border-blue-500 focus:outline-none"
              />

              {inputText && (
                <div className="bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-700 dark:to-gray-800 p-6 rounded-xl shadow-md mb-4">
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="text-xl font-semibold">📄 Original Text</h3>
                    <div className="flex gap-2">
                      <button
                        onClick={() => speak(translatedText, language)}
                        className="bg-green-500 px-3 py-2 rounded-lg text-white hover:bg-green-600 transition text-sm"
                      >
                        🔊 Read
                      </button>
                      <button
                        onClick={() => copyToClipboard(translatedText)}
                        className="bg-gray-500 px-3 py-2 rounded-lg text-white hover:bg-gray-600 transition text-sm"
                      >
                        📋 Copy
                      </button>
                      <button
                        onClick={stopSpeech}
                        className="bg-red-500 px-3 py-2 rounded-lg text-white hover:bg-red-600 transition text-sm"
                      >
                        ⏹ Stop
                      </button>
                    </div>
                  </div>
                  <div className="text-gray-700 dark:text-gray-200">
                    {translatedText.split('\n').map((para, i) => (
                      <p key={i} className="mb-3">{para}</p>
                    ))}
                  </div>
                </div>
              )}

              {summary && (
                <div className="bg-gradient-to-br from-blue-50 to-purple-50 dark:from-blue-900 dark:to-purple-900 p-6 rounded-xl shadow-md">
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="text-xl font-semibold">✨ Summary</h3>
                    <div className="flex gap-2">
                      <button
                        onClick={() => speak(translatedSummary, language)}
                        className="bg-purple-500 px-3 py-2 rounded-lg text-white hover:bg-purple-600 transition text-sm"
                      >
                        🔊 Read
                      </button>
                      <button
                        onClick={() => copyToClipboard(translatedSummary)}
                        className="bg-gray-500 px-3 py-2 rounded-lg text-white hover:bg-gray-600 transition text-sm"
                      >
                        📋 Copy
                      </button>
                      <button
                        onClick={stopSpeech}
                        className="bg-red-500 px-3 py-2 rounded-lg text-white hover:bg-red-600 transition text-sm"
                      >
                        ⏹ Stop
                      </button>
                    </div>
                  </div>
                  <p className="text-gray-700 dark:text-gray-100 leading-relaxed">{translatedSummary}</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Sentiment Analyzer Tab */}
        {activeTab === 'sentiment' && (
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
              <h2 className="text-2xl font-bold mb-4 text-purple-600 dark:text-purple-400">
                Sentiment Analyzer
              </h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Discover the emotional tone behind any text. Our AI detects positive, negative, and neutral sentiments.
              </p>

              <textarea
                value={sentimentText}
                onChange={(e) => setSentimentText(e.target.value)}
                placeholder="Type or paste your text here..."
                className="w-full h-40 p-4 border-2 rounded-xl bg-gray-50 dark:bg-gray-700 dark:text-white border-gray-300 dark:border-gray-600 resize-none mb-4 focus:border-purple-500 focus:outline-none"
              />

              <div className="flex gap-3 mb-4">
                <button
                  onClick={handleAnalyzeSentiment}
                  disabled={analyzingsentiment}
                  className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 rounded-xl font-semibold hover:shadow-lg transition disabled:opacity-50"
                >
                  {analyzingsentiment ? '⏳ Analyzing...' : '🔍 Analyze Sentiment'}
                </button>
                {sentimentText && (
                  <>
                    <button
                      onClick={() => speak(sentimentText, 'en')}
                      className="bg-green-500 px-4 py-3 rounded-xl text-white hover:bg-green-600 transition"
                      title="Read Text"
                    >
                      🔊
                    </button>
                    <button
                      onClick={() => copyToClipboard(sentimentText)}
                      className="bg-gray-500 px-4 py-3 rounded-xl text-white hover:bg-gray-600 transition"
                      title="Copy Text"
                    >
                      📋
                    </button>
                    <button
                      onClick={stopSpeech}
                      className="bg-red-500 px-4 py-3 rounded-xl text-white hover:bg-red-600 transition"
                      title="Stop Speech"
                    >
                      ⏹
                    </button>
                  </>
                )}
              </div>

              {sentimentResult && (
                <div className="mt-6 space-y-4">
                  <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900 dark:to-pink-900 p-6 rounded-xl">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-4">
                        <div className={`text-5xl ${
                          sentimentResult.sentimentClass === 'positive' ? 'bg-cyan-100 dark:bg-cyan-900' :
                          sentimentResult.sentimentClass === 'negative' ? 'bg-purple-100 dark:bg-purple-900' :
                          'bg-pink-100 dark:bg-pink-900'
                        } w-20 h-20 rounded-full flex items-center justify-center`}>
                          {sentimentResult.icon}
                        </div>
                        <div>
                          <div className={`text-3xl font-bold ${
                            sentimentResult.sentimentClass === 'positive' ? 'text-cyan-600 dark:text-cyan-400' :
                            sentimentResult.sentimentClass === 'negative' ? 'text-purple-600 dark:text-purple-400' :
                            'text-pink-600 dark:text-pink-400'
                          }`}>
                            {sentimentResult.sentiment}
                          </div>
                          <div className="text-lg text-gray-600 dark:text-gray-300">
                            Confidence: {sentimentResult.confidence}%
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="flex h-6 rounded-full overflow-hidden shadow-inner mb-4">
                      <div 
                        className="bg-cyan-400 dark:bg-cyan-600 transition-all duration-500"
                        style={{ width: `${sentimentResult.positive_percent}%` }}
                      ></div>
                      <div 
                        className="bg-pink-400 dark:bg-pink-600 transition-all duration-500"
                        style={{ width: `${sentimentResult.neutral_percent}%` }}
                      ></div>
                      <div 
                        className="bg-purple-400 dark:bg-purple-600 transition-all duration-500"
                        style={{ width: `${sentimentResult.negative_percent}%` }}
                      ></div>
                    </div>

                    <div className="flex justify-around text-sm">
                      <div className="text-center">
                        <div className="font-semibold text-cyan-600 dark:text-cyan-400">Positive</div>
                        <div>{sentimentResult.positive_percent}%</div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold text-pink-600 dark:text-pink-400">Neutral</div>
                        <div>{sentimentResult.neutral_percent}%</div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold text-purple-600 dark:text-purple-400">Negative</div>
                        <div>{sentimentResult.negative_percent}%</div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gray-50 dark:bg-gray-700 p-6 rounded-xl border-2 border-gray-200 dark:border-gray-600">
                    <div className="flex justify-between items-center mb-3">
                      <h4 className="font-semibold text-lg">Highlighted Text:</h4>
                      <div className="flex gap-2">
                        <button
                          onClick={() => speak(sentimentText, 'en')}
                          className="bg-green-500 px-3 py-2 rounded-lg text-white hover:bg-green-600 transition text-sm"
                        >
                          🔊 Read
                        </button>
                        <button
                          onClick={() => copyToClipboard(sentimentText)}
                          className="bg-gray-500 px-3 py-2 rounded-lg text-white hover:bg-gray-600 transition text-sm"
                        >
                          📋 Copy
                        </button>
                        <button
                          onClick={stopSpeech}
                          className="bg-red-500 px-3 py-2 rounded-lg text-white hover:bg-red-600 transition text-sm"
                        >
                          ⏹ Stop
                        </button>
                      </div>
                    </div>
                    <div 
                      className="leading-relaxed"
                      dangerouslySetInnerHTML={{ __html: sentimentResult.highlighted }}
                    />
                  </div>
                </div>
              )}

              {/* Example Cards */}
              <div className="mt-8">
                <h3 className="text-xl font-semibold mb-4">💡 Try These Examples:</h3>
                <div className="grid md:grid-cols-3 gap-4">
                  {exampleTexts.map((example, idx) => (
                    <div
                      key={idx}
                      onClick={() => {
                        setSentimentText(example.text);
                        setTimeout(() => handleAnalyzeSentiment(), 100);
                      }}
                      className="bg-white dark:bg-gray-700 p-4 rounded-xl shadow-md hover:shadow-lg transition cursor-pointer border-2 border-gray-200 dark:border-gray-600 hover:border-purple-400"
                    >
                      <h4 className={`font-semibold mb-2 ${
                        example.type === 'positive' ? 'text-cyan-600 dark:text-cyan-400' :
                        example.type === 'negative' ? 'text-purple-600 dark:text-purple-400' :
                        'text-pink-600 dark:text-pink-400'
                      }`}>
                        {example.type === 'positive' ? '😊 Positive' :
                         example.type === 'negative' ? '😞 Negative' : '😐 Neutral'}
                      </h4>
                      <p className="text-sm text-gray-600 dark:text-gray-300 italic">
                        {example.text.substring(0, 80)}...
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Humanizer Tab */}
        {activeTab === 'humanizer' && (
          <div className="space-y-6">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-6">
              <h2 className="text-2xl font-bold mb-4 text-green-600 dark:text-green-400">
                Text Humanizer
              </h2>
              <p className="text-gray-600 dark:text-gray-300 mb-6">
                Transform robotic or AI-generated text into natural, conversational language. Perfect for making formal content more readable and engaging.
              </p>

              <textarea
                value={humanizerText}
                onChange={(e) => setHumanizerText(e.target.value)}
                placeholder="Paste your AI-generated or formal text here..."
                className="w-full h-40 p-4 border-2 rounded-xl bg-gray-50 dark:bg-gray-700 dark:text-white border-gray-300 dark:border-gray-600 resize-none mb-4 focus:border-green-500 focus:outline-none"
              />

              <div className="flex gap-3 mb-4">
                <button
                  onClick={handleHumanize}
                  disabled={humanizingText}
                  className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 text-white py-3 rounded-xl font-semibold hover:shadow-lg transition disabled:opacity-50"
                >
                  {humanizingText ? '⏳ Humanizing...' : '👤 Humanize Text'}
                </button>
                {humanizerText && (
                  <>
                    <button
                      onClick={() => speak(humanizerText, 'en')}
                      className="bg-green-500 px-4 py-3 rounded-xl text-white hover:bg-green-600 transition"
                      title="Read Text"
                    >
                      🔊
                    </button>
                    <button
                      onClick={() => copyToClipboard(humanizerText)}
                      className="bg-gray-500 px-4 py-3 rounded-xl text-white hover:bg-gray-600 transition"
                      title="Copy Text"
                    >
                      📋
                    </button>
                    <button
                      onClick={stopSpeech}
                      className="bg-red-500 px-4 py-3 rounded-xl text-white hover:bg-red-600 transition"
                      title="Stop Speech"
                    >
                      ⏹
                    </button>
                  </>
                )}
              </div>

              {humanizerText && (
                <div className="bg-gradient-to-br from-gray-50 to-green-50 dark:from-gray-700 dark:to-gray-800 p-6 rounded-xl shadow-md mb-4">
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="text-xl font-semibold">📝 Original Text</h3>
                    <div className="flex gap-2">
                      <button
                        onClick={() => speak(humanizerText, 'en')}
                        className="bg-green-500 px-3 py-2 rounded-lg text-white hover:bg-green-600 transition text-sm"
                      >
                        🔊 Read
                      </button>
                      <button
                        onClick={() => copyToClipboard(humanizerText)}
                        className="bg-gray-500 px-3 py-2 rounded-lg text-white hover:bg-gray-600 transition text-sm"
                      >
                        📋 Copy
                      </button>
                      <button
                        onClick={stopSpeech}
                        className="bg-red-500 px-3 py-2 rounded-lg text-white hover:bg-red-600 transition text-sm"
                      >
                        ⏹ Stop
                      </button>
                    </div>
                  </div>
                  <div className="text-gray-700 dark:text-gray-200">
                    {humanizerText.split('\n').map((para, i) => (
                      <p key={i} className="mb-3">{para}</p>
                    ))}
                  </div>
                </div>
              )}

              {humanizedText && (
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900 dark:to-emerald-900 p-6 rounded-xl shadow-md">
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="text-xl font-semibold">✨ Humanized Text</h3>
                    <div className="flex gap-2">
                      <button
                        onClick={() => speak(humanizedText, 'en')}
                        className="bg-green-500 px-3 py-2 rounded-lg text-white hover:bg-green-600 transition text-sm"
                      >
                        🔊 Read
                      </button>
                      <button
                        onClick={() => copyToClipboard(humanizedText)}
                        className="bg-gray-500 px-3 py-2 rounded-lg text-white hover:bg-gray-600 transition text-sm"
                      >
                        📋 Copy
                      </button>
                      <button
                        onClick={stopSpeech}
                        className="bg-red-500 px-3 py-2 rounded-lg text-white hover:bg-red-600 transition text-sm"
                      >
                        ⏹ Stop
                      </button>
                    </div>
                  </div>
                  <p className="text-gray-700 dark:text-gray-100 leading-relaxed whitespace-pre-wrap">{humanizedText}</p>
                </div>
              )}

              {/* Example Cards */}
              <div className="mt-8">
                <h3 className="text-xl font-semibold mb-4">💡 Example Transformations:</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-xl border-2 border-gray-200 dark:border-gray-600">
                    <div className="font-semibold text-red-600 dark:text-red-400 mb-2">❌ Robotic:</div>
                    <p className="text-sm text-gray-600 dark:text-gray-300 italic">
                      "It is important to note that the utilization of this methodology will facilitate the achievement of optimal results."
                    </p>
                  </div>
                  <div className="bg-green-50 dark:bg-green-900 p-4 rounded-xl border-2 border-green-200 dark:border-green-600">
                    <div className="font-semibold text-green-600 dark:text-green-400 mb-2">✅ Human-like:</div>
                    <p className="text-sm text-gray-600 dark:text-gray-300 italic">
                      "Keep in mind that using this method will help you get the best results."
                    </p>
                  </div>
                  <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-xl border-2 border-gray-200 dark:border-gray-600">
                    <div className="font-semibold text-red-600 dark:text-red-400 mb-2">❌ Formal:</div>
                    <p className="text-sm text-gray-600 dark:text-gray-300 italic">
                      "In accordance with our predetermined protocol, all personnel must ascertain compliance prior to project commencement."
                    </p>
                  </div>
                  <div className="bg-green-50 dark:bg-green-900 p-4 rounded-xl border-2 border-green-200 dark:border-green-600">
                    <div className="font-semibold text-green-600 dark:text-green-400 mb-2">✅ Natural:</div>
                    <p className="text-sm text-gray-600 dark:text-gray-300 italic">
                      "Following our plan, everyone should check that they're ready before starting the project."
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="text-center py-6 mt-12 border-t border-gray-300 dark:border-gray-700">
        <p className="text-gray-600 dark:text-gray-400">
          © 2025 AI Text Analysis Suite | Powered by Advanced AI
        </p>
      </footer>
    </div>
  );
}

export default App;