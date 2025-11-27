import { useState, useEffect } from 'react';

function App() {
  const [activeTab, setActiveTab] = useState('summarizer');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState('en');
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');
  const [translatedSummary, setTranslatedSummary] = useState('');
  const [darkMode, setDarkMode] = useState(false);
  
  // Sentiment Analysis States
  const [sentimentText, setSentimentText] = useState('');
  const [sentimentResult, setSentimentResult] = useState(null);
  const [analyzingsentiment, setAnalyzingsentiment] = useState(false);

  // Mock summarizer function
  const summarizeText = async (text) => {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
    if (sentences.length <= 2) return text;
    
    const summary = sentences.slice(0, Math.ceil(sentences.length / 3)).join(' ');
    return summary;
  };

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

  const handleSummarize = async () => {
    stopSpeech();
    if (!inputText.trim()) {
      alert("⚠️ Please paste or type some text first.");
      return;
    }
    setLoading(true);
    const result = await summarizeText(inputText);
    setSummary(result);
    const translated = await translateText(result, language);
    setTranslatedSummary(translated);
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
    setDarkMode(!darkMode);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      alert('📋 Copied to clipboard!');
    });
  };

  // Sentiment Analysis Functions
  const analyzeSentiment = () => {
    if (!sentimentText.trim()) {
      alert('Please enter some text to analyze.');
      return;
    }
    
    setAnalyzingsentiment(true);
    
    setTimeout(() => {
      const result = getSentimentAnalysis(sentimentText);
      setSentimentResult(result);
      setAnalyzingsentiment(false);
    }, 800);
  };

  const getSentimentAnalysis = (text) => {
    const positiveWords = {
      'love': 3, 'great': 2, 'excellent': 3, 'fantastic': 3, 'amazing': 3,
      'wonderful': 3, 'perfect': 3, 'best': 2, 'awesome': 3, 'superb': 3,
      'outstanding': 3, 'happy': 2, 'satisfied': 2, 'recommend': 2, 'pleased': 2,
      'delighted': 3, 'impressed': 2, 'good': 1, 'nice': 1, 'easy': 1, 'fast': 1
    };
    
    const negativeWords = {
      'hate': 3, 'terrible': 3, 'awful': 3, 'poor': 2, 'bad': 2, 'horrible': 3,
      'disappointed': 3, 'worst': 3, 'regret': 2, 'avoid': 2, 'problem': 1,
      'issue': 1, 'broken': 2, 'frustrated': 2, 'unhappy': 2, 'dislike': 2,
      'angry': 2, 'difficult': 1, 'slow': 1, 'expensive': 1
    };
    
    let positiveScore = 0;
    let negativeScore = 0;
    
    const words = text.toLowerCase().match(/\b(\w+)\b/g) || [];
    
    words.forEach(word => {
      const cleanWord = word.replace(/[^\w\s]|_/g, "").replace(/\s+/g, " ");
      if (positiveWords[cleanWord]) {
        positiveScore += positiveWords[cleanWord];
      } else if (negativeWords[cleanWord]) {
        negativeScore += negativeWords[cleanWord];
      }
    });
    
    const totalScore = positiveScore + Math.abs(negativeScore);
    let positivePercent = 0;
    let negativePercent = 0;
    let neutralPercent = 0;
    
    if (totalScore > 0) {
      positivePercent = Math.round((positiveScore / totalScore) * 85);
      negativePercent = Math.round((Math.abs(negativeScore) / totalScore) * 85);
      neutralPercent = 100 - (positivePercent + negativePercent);
    } else {
      neutralPercent = 100;
    }
    
    positivePercent = Math.min(95, positivePercent + 5);
    negativePercent = Math.min(95, negativeScore > 0 ? negativePercent + 5 : negativePercent);
    neutralPercent = 100 - positivePercent - negativePercent;
    
    let sentiment, sentimentClass, icon, confidence;
    
    if (positiveScore > negativeScore && positiveScore > 0) {
      sentiment = 'Positive';
      sentimentClass = 'positive';
      icon = '😊';
      confidence = Math.min(98, positivePercent + 10);
    } else if (negativeScore > positiveScore && negativeScore > 0) {
      sentiment = 'Negative';
      sentimentClass = 'negative';
      icon = '😞';
      confidence = Math.min(98, negativePercent + 10);
    } else {
      sentiment = 'Neutral';
      sentimentClass = 'neutral';
      icon = '😐';
      confidence = Math.max(50, neutralPercent);
    }
    
    let highlighted = text;
    Object.keys(positiveWords).forEach(word => {
      const regex = new RegExp(`\\b${word}\\b`, 'gi');
      highlighted = highlighted.replace(regex, match => `<span class="bg-cyan-200 dark:bg-cyan-900 px-1 rounded">${match}</span>`);
    });
    
    Object.keys(negativeWords).forEach(word => {
      const regex = new RegExp(`\\b${word}\\b`, 'gi');
      highlighted = highlighted.replace(regex, match => `<span class="bg-purple-200 dark:bg-purple-900 px-1 rounded">${match}</span>`);
    });
    
    return {
      sentiment,
      confidence,
      sentimentClass,
      icon,
      highlighted,
      positivePercent,
      negativePercent,
      neutralPercent
    };
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
        <div className="flex gap-4 mb-6 bg-white dark:bg-gray-800 p-2 rounded-xl shadow-md">
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
                  onClick={analyzeSentiment}
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
                        style={{ width: `${sentimentResult.positivePercent}%` }}
                      ></div>
                      <div 
                        className="bg-pink-400 dark:bg-pink-600 transition-all duration-500"
                        style={{ width: `${sentimentResult.neutralPercent}%` }}
                      ></div>
                      <div 
                        className="bg-purple-400 dark:bg-purple-600 transition-all duration-500"
                        style={{ width: `${sentimentResult.negativePercent}%` }}
                      ></div>
                    </div>

                    <div className="flex justify-around text-sm">
                      <div className="text-center">
                        <div className="font-semibold text-cyan-600 dark:text-cyan-400">Positive</div>
                        <div>{sentimentResult.positivePercent}%</div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold text-pink-600 dark:text-pink-400">Neutral</div>
                        <div>{sentimentResult.neutralPercent}%</div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold text-purple-600 dark:text-purple-400">Negative</div>
                        <div>{sentimentResult.negativePercent}%</div>
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
                        setTimeout(() => analyzeSentiment(), 100);
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