import React, { useState, useEffect, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import IconLogo from './IconLogo';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL;

export default function ChatApp() {
  // Session and message states
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [ingestedFiles, setIngestedFiles] = useState([]);
  const [activeViewFile, setActiveViewFile] = useState(null);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);

  // Knowledge ingestion states
  const [dataType, setDataType] = useState('text');
  const [segmentTitle, setSegmentTitle] = useState('');
  const [textContent, setTextContent] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');

  // UI states
  const [activeTab, setActiveTab] = useState('chat');
  const [maxNodes, setMaxNodes] = useState(20);
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [communities, setCommunities] = useState([]); 
  const [graphLoading, setGraphLoading] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [activeMobileSection, setActiveMobileSection] = useState('chat');

  // Graph refs and dimensions
  const graphRef = useRef();
  const graphContainerRef = useRef();
  const [graphDimensions, setGraphDimensions] = useState({ width: 400, height: 400 });
  const messagesContainerRef = useRef(null);

  const getAuthHeaders = () => {
    const token = localStorage.getItem('token');
    return { 'Authorization': `Token ${token}` };
  };

  const handleSignOut = () => {
    localStorage.removeItem('token');
    window.location.href = '/login';
  };

  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  }, [messages, loading, isSending]);

  useEffect(() => {
    const updateDimensions = () => {
      if (graphContainerRef.current) {
        const width = graphContainerRef.current.clientWidth;
        const height = graphContainerRef.current.clientHeight;
        if (width > 0 && height > 0) {
          setGraphDimensions({ width, height });
        }
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [activeTab, graphData, activeMobileSection]);

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const response = await axios.get(`${API_URL}/chats/`, { headers: getAuthHeaders() });
        setSessions(response.data);
        if (response.data.length > 0) {
          const firstSessionId = response.data[0].id;
          setCurrentSessionId(firstSessionId);
          fetchSessionData(firstSessionId);
        }
      } catch (error) {
        console.error("Error fetching chat sessions:", error);
      }
    };
    fetchSessions();
  }, []);

  const fetchGraphData = async (sessionId, nodeLimit = maxNodes) => {
    if (!sessionId) return;
    setGraphLoading(true);
    try {
      const response = await axios.get(`${API_URL}/chats/${sessionId}/graph/?max_nodes=${nodeLimit}`, {
        headers: getAuthHeaders(),
      });

      const connections = response.data.connections || [];
      const uniqueNodes = new Set();

      const links = connections.map(conn => {
        uniqueNodes.add(conn.source);
        uniqueNodes.add(conn.target);
        return { source: conn.source, target: conn.target, val: conn.weight || 1 };
      });

      const nodes = Array.from(uniqueNodes).map(nodeId => ({ id: nodeId }));
      setGraphData({ nodes, links });
      
      setCommunities(response.data.communities || []);

    } catch (error) {
      console.error("Error fetching graph data:", error);
    } finally {
      setGraphLoading(false);
    }
  };

  useEffect(() => {
    if (currentSessionId) {
      fetchGraphData(currentSessionId, maxNodes);
    }
  }, [currentSessionId, maxNodes]);

  const fetchSessionData = async (sessionId) => {
    setLoading(true);
    try {
      const msgResponse = await axios.get(`${API_URL}/chats/${sessionId}/messages/`, { headers: getAuthHeaders() });
      setMessages(msgResponse.data);

      const sessionResponse = await axios.get(`${API_URL}/chats/${sessionId}/`, { headers: getAuthHeaders() });
      setIngestedFiles(sessionResponse.data.files || []);
      setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, ...sessionResponse.data } : s));
    } catch (error) {
      console.error("Error fetching session details:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectSession = (sessionId) => {
    setCurrentSessionId(sessionId);
    fetchSessionData(sessionId);
    setIsSidebarOpen(false);
  };

  const handleCreateSession = async () => {
    try {
      const response = await axios.post(`${API_URL}/chats/`, { title: "New Conversation" }, { headers: getAuthHeaders() });
      setSessions([response.data, ...sessions]);
      setCurrentSessionId(response.data.id);
      setMessages([]);
      setIngestedFiles([]);
      setGraphData({ nodes: [], links: [] });
      setCommunities([]); 
      setActiveTab('chat');
      setIsSidebarOpen(false);
    } catch (error) {
      console.error("Failed to initialize session:", error);
    }
  };

  const handleDeleteSession = async (e, sessionId) => {
    e.stopPropagation();
    try {
      await axios.delete(`${API_URL}/chats/${sessionId}/`, { headers: getAuthHeaders() });
      const updatedSessions = sessions.filter(s => s.id !== sessionId);
      setSessions(updatedSessions);

      if (currentSessionId === sessionId) {
        if (updatedSessions.length > 0) {
          const fallbackId = updatedSessions[0].id;
          setCurrentSessionId(fallbackId);
          fetchSessionData(fallbackId);
        } else {
          setCurrentSessionId(null);
          setMessages([]);
          setIngestedFiles([]);
          setGraphData({ nodes: [], links: [] });
          setCommunities([]);
        }
      }
    } catch (error) {
      console.error("Failed to delete session:", error);
      alert("Error deleting session.");
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || !currentSessionId || isSending) return;

    const messageContentSnapshot = inputMessage.trim();
    const userMessagePayload = { role: 'user', content: messageContentSnapshot, metadata: {} };

    setMessages(prev => [...prev, userMessagePayload]);
    setInputMessage('');
    setIsSending(true);

    try {
      const response = await axios.post(
        `${API_URL}/chats/${currentSessionId}/messages/`,
        userMessagePayload,
        { headers: getAuthHeaders() }
      );

      if (response.data?.role === 'assistant') {
        setMessages(prev => [...prev, response.data]);
        const sessionRes = await axios.get(`${API_URL}/chats/${currentSessionId}/`, { headers: getAuthHeaders() });
        if (sessionRes.data.title) {
          setSessions(prev => prev.map(s => s.id === currentSessionId ? { ...s, title: sessionRes.data.title } : s));
        }
      } else {
        await fetchSessionData(currentSessionId);
      }
    } catch (error) {
      console.error("Transmission error:", error);
    } finally {
      setIsSending(false);
    }
  };

  const handleIngestData = async (e) => {
    e.preventDefault();
    if (isIngesting || !currentSessionId) return;

    let finalContent = '';
    let finalTitle = segmentTitle || "Pasted Text";

    if (dataType === 'text') {
      if (!textContent.trim()) { setUploadStatus('Please enter text content.'); return; }
      finalContent = textContent;
      setIsIngesting(true);
      setUploadStatus('Processing Knowledge Ingestion...');
      await sendToGraphBackend(finalTitle, finalContent);
    } else {
      if (!selectedFile) { setUploadStatus('Please select a .txt file first.'); return; }
      setIsIngesting(true);
      setUploadStatus('Processing Knowledge Ingestion...');
      const reader = new FileReader();
      reader.onload = async (event) => {
        finalContent = event.target.result;
        if (!segmentTitle) finalTitle = selectedFile.name;
        await sendToGraphBackend(finalTitle, finalContent);
      };
      reader.onerror = () => { setUploadStatus('Error reading file.'); setIsIngesting(false); };
      reader.readAsText(selectedFile);
    }
  };

  const sendToGraphBackend = async (title, content) => {
    try {
      await axios.post(
        `${API_URL}/chats/${currentSessionId}/messages/`,
        { title, content, is_ingestion: true },
        { headers: getAuthHeaders() }
      );
      setUploadStatus('Success! Content pushed to Knowledge Graph.');
      setSegmentTitle(''); setTextContent(''); setSelectedFile(null);
      if (document.getElementById('fileInput')) document.getElementById('fileInput').value = '';
      await fetchSessionData(currentSessionId);
      
      fetchGraphData(currentSessionId, maxNodes);
      
    } catch (error) {
      console.error("Ingestion failed:", error);
      setUploadStatus('Error parsing data segment into Graph.');
    } finally {
      setIsIngesting(false);
    }
  };

  return (
    <div className="flex h-screen w-screen font-sans bg-slate-50 overflow-hidden">
      {isSidebarOpen && (
        <div className="fixed inset-0 bg-black/50 z-40 lg:hidden transition-opacity" onClick={() => setIsSidebarOpen(false)} />
      )}

      {/* Sidebar */}
      <div className={`fixed lg:relative inset-y-0 left-0 w-72 border-r border-slate-200 bg-white flex flex-col z-50 transition-transform duration-300 ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}`}>
        <div className="h-16 px-6 border-b border-slate-100 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-500/30">
              <IconLogo />
            </div>
            <span className="font-bold text-xl tracking-tight">GraphRAG Studio</span>
          </div>
        </div>

        <button onClick={handleCreateSession} className="mx-6 mt-4 mb-6 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3.5 rounded-2xl shadow-lg transition">
          + New Session
        </button>

        <h4 className="px-6 text-xs font-bold uppercase tracking-widest text-slate-500 mb-3">Active Threads</h4>
        <div className="flex-1 overflow-y-auto px-4 space-y-1 pb-6">
          {sessions.map((session) => (
            <div key={session.id} onClick={() => handleSelectSession(session.id)} className={`group flex items-center justify-between px-4 py-3 rounded-2xl cursor-pointer transition-all ${currentSessionId === session.id ? 'bg-slate-100' : 'hover:bg-slate-50'}`}>
              <div className="flex items-center gap-3 overflow-hidden flex-1">
                <span className="text-lg">💬</span>
                <span className="truncate font-medium text-sm text-slate-800">{session.title || "Untitled Conversation"}</span>
              </div>
              <button onClick={(e) => handleDeleteSession(e, session.id)} className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-500 p-1 rounded-lg hover:bg-red-50 transition">✕</button>
            </div>
          ))}
        </div>

        <div className="p-6 border-t border-slate-100 shrink-0">
          <button onClick={handleSignOut} className="w-full bg-red-500 hover:bg-red-600 text-white font-semibold py-3.5 rounded-2xl shadow-lg shadow-red-500/30 transition flex items-center justify-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M3 3a1 1 0 00-1 1v12a1 1 0 102 0V4a1 1 0 00-1-1zm10.293 9.293a1 1 0 001.414 1.414l3-3a1 1 0 000-1.414l-3-3a1 1 0 10-1.414 1.414L14.586 9H7a1 1 0 100 2h7.586l-1.293 1.293z" clipRule="evenodd" />
            </svg>
            Sign Out
          </button>
        </div>
      </div>

      {/* Main workspace */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <div className="h-16 border-b border-slate-100 bg-white flex items-center px-4 sm:px-6 justify-between shrink-0">
          <div className="flex items-center gap-2 sm:gap-4">
            <button onClick={() => setIsSidebarOpen(true)} className="lg:hidden p-3 rounded-xl hover:bg-slate-100 text-xl">☰</button>
            <div className="bg-slate-100 p-1 rounded-3xl flex">
              <button onClick={() => { setActiveTab('chat'); setActiveMobileSection('chat'); }} className={`px-3 sm:px-6 py-2 rounded-3xl text-xs sm:text-sm font-semibold transition whitespace-nowrap ${activeTab === 'chat' && activeMobileSection !== 'knowledge' ? 'bg-white shadow text-slate-900' : 'text-slate-600'}`}>Conversation</button>
              <button onClick={() => { setActiveTab('graph'); setActiveMobileSection('chat'); }} disabled={!currentSessionId} className={`px-3 sm:px-6 py-2 rounded-3xl text-xs sm:text-sm font-semibold transition whitespace-nowrap ${activeTab === 'graph' && activeMobileSection !== 'knowledge' ? 'bg-white shadow text-slate-900' : 'text-slate-600'} ${!currentSessionId && 'opacity-50 cursor-not-allowed'}`}>Visual Graph</button>
              <button onClick={() => setActiveMobileSection(prev => prev === 'knowledge' ? 'chat' : 'knowledge')} className={`lg:hidden px-3 sm:px-6 py-2 rounded-3xl text-xs sm:text-sm font-semibold transition whitespace-nowrap ${activeMobileSection === 'knowledge' ? 'bg-white shadow text-slate-900' : 'text-slate-600'}`}>Knowledge</button>
            </div>
          </div>
        </div>

        <div className="flex-1 flex overflow-hidden">
          <div className={`flex-1 flex-col min-w-0 ${activeMobileSection === 'knowledge' ? 'hidden' : 'flex'} lg:flex`}>
            {activeTab === 'chat' ? (
              <>
                <div ref={messagesContainerRef} className="flex-1 overflow-y-auto p-6 bg-slate-50 space-y-6">
                  {loading ? (
                    <div className="text-center text-slate-500 py-12">Syncing workspace architecture...</div>
                  ) : messages.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-slate-400 pt-20">
                      <span className="text-7xl mb-6">🧠</span>
                      <p className="font-semibold text-lg">Knowledge Graph Ready</p>
                    </div>
                  ) : (
                    <>
                      {messages.map((msg, index) => (
                        <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                          <div className={`max-w-[80%] px-6 py-4 rounded-3xl text-[15px] leading-relaxed shadow-sm break-words ${msg.role === 'user' ? 'bg-indigo-600 text-white rounded-br-none whitespace-pre-wrap' : 'bg-white border border-slate-200 rounded-bl-none prose prose-sm max-w-none'}`}>
                            {msg.role === 'assistant' ? (
                              <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ p: ({node, ...props}) => <p className="mb-3 last:mb-0 text-slate-700" {...props} />, ul: ({node, ...props}) => <ul className="list-disc pl-5 mb-3 space-y-1.5 text-slate-700" {...props} />, ol: ({node, ...props}) => <ol className="list-decimal pl-5 mb-3 space-y-1.5 text-slate-700" {...props} />, li: ({node, ...props}) => <li className="pl-1" {...props} />, strong: ({node, ...props}) => <strong className="font-bold text-slate-900" {...props} />, em: ({node, ...props}) => <em className="italic text-slate-600" {...props} />, code: ({node, inline, ...props}) => inline ? <code className="bg-slate-100 text-indigo-600 px-1.5 py-0.5 rounded text-[13px] font-mono font-medium" {...props} /> : <pre className="bg-slate-900 text-slate-100 p-4 rounded-xl overflow-x-auto my-3 border border-slate-700 shadow-inner"><code className="text-[13px] font-mono leading-relaxed" {...props} /></pre>, a: ({node, ...props}) => <a className="text-indigo-600 underline decoration-indigo-300 hover:text-indigo-800 transition font-medium" target="_blank" rel="noopener noreferrer" {...props} />, h1: ({node, ...props}) => <h1 className="text-xl font-bold mb-3 text-slate-900" {...props} />, h2: ({node, ...props}) => <h2 className="text-lg font-bold mb-2 text-slate-900" {...props} />, h3: ({node, ...props}) => <h3 className="text-base font-bold mb-2 text-slate-800" {...props} />, blockquote: ({node, ...props}) => <blockquote className="border-l-4 border-indigo-300 pl-4 italic text-slate-600 my-3" {...props} />, hr: ({node, ...props}) => <hr className="my-4 border-slate-200" {...props} />, table: ({node, ...props}) => <div className="overflow-x-auto my-3 rounded-lg border border-slate-200"><table className="min-w-full text-sm text-left text-slate-700" {...props} /></div>, th: ({node, ...props}) => <th className="px-4 py-2 bg-slate-50 font-semibold text-slate-900 border-b border-slate-200" {...props} />, td: ({node, ...props}) => <td className="px-4 py-2 border-b border-slate-100" {...props} />, }}>
                                {msg.content}
                              </ReactMarkdown>
                            ) : (
                              msg.content
                            )}
                          </div>
                        </div>
                      ))}
                      {isSending && (
                        <div className="flex justify-start">
                          <div className="bg-white border border-slate-200 rounded-3xl rounded-bl-none px-6 py-4 shadow-sm flex items-center gap-1.5">
                            <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                            <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                            <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce"></div>
                          </div>
                        </div>
                      )}
                    </>
                  )}
                </div>

                <div className="p-6 border-t border-slate-200 bg-white shrink-0">
                  <form onSubmit={handleSendMessage} className="flex gap-3">
                    <input type="text" value={inputMessage} onChange={(e) => setInputMessage(e.target.value)} placeholder={currentSessionId ? "Ask a question about your knowledge graph..." : "Select or create a session"} disabled={!currentSessionId || isSending} className="flex-1 min-w-0 px-4 sm:px-6 py-4 border border-slate-200 rounded-2xl focus:outline-none focus:border-indigo-300 disabled:bg-slate-100 text-sm" />
                    <button type="submit" disabled={!currentSessionId || !inputMessage.trim() || isSending} className="shrink-0 w-32 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white font-semibold rounded-2xl transition flex items-center justify-center gap-2">
                      {isSending ? (<><svg className="animate-spin h-4 w-4 text-white shrink-0" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Processing...</>) : 'Send'}
                    </button>
                  </form>
                </div>
              </>
            ) : (
              <div ref={graphContainerRef} className="flex-1 bg-slate-950 relative overflow-hidden">
                {graphLoading ? (
                  <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 z-10">
                    <div className="text-slate-400 text-lg">Extracting topological matrices...</div>
                  </div>
                ) : graphData.nodes.length === 0 ? (
                  <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-center px-8">
                    No relational structures mapped yet.<br />Ingest documents to generate the graph.
                  </div>
                ) : (
                  <ForceGraph2D
                    ref={graphRef}
                    graphData={graphData}
                    width={graphDimensions.width}
                    height={graphDimensions.height}
                    backgroundColor="#0f172a"
                    nodeColor={() => '#818cf8'}
                    nodeLabel="id"
                    nodeRelSize={6}
                    linkColor={() => 'rgba(255,255,255,0.08)'}
                    linkWidth={link => Math.min(link.val * 1.5, 5)}
                    cooldownTicks={100}
                    nodeCanvasObject={(node, ctx, globalScale) => {
                      const label = node.id;
                      const fontSize = 11 / globalScale;
                      ctx.font = `500 ${fontSize}px Inter, sans-serif`;
                      ctx.fillStyle = 'rgba(255,255,255,0.75)';
                      ctx.fillText(label, node.x + 8, node.y + 3);
                      ctx.beginPath();
                      ctx.arc(node.x, node.y, 5, 0, 2 * Math.PI, false);
                      ctx.fillStyle = '#6366f1';
                      ctx.fill();
                      ctx.lineWidth = 1.5 / globalScale;
                      ctx.strokeStyle = '#ffffff';
                      ctx.stroke();
                    }}
                  />
                )}
              </div>
            )}
          </div>

          {/* Knowledge panel */}
          <div className={`w-full lg:w-96 border-l border-slate-200 bg-white flex flex-col overflow-hidden ${activeMobileSection === 'knowledge' ? 'flex' : 'hidden'} lg:flex`}>
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              <div className="bg-slate-50 border border-slate-200 rounded-2xl p-5">
                <label className="flex justify-between text-sm font-semibold text-slate-600 mb-3">
                  Visual Node Limit
                  <span className="text-indigo-600 font-bold">{maxNodes}</span>
                </label>
                <input type="range" min="5" max="500" step="5" value={maxNodes} onChange={(e) => setMaxNodes(parseInt(e.target.value))} className="w-full accent-indigo-600" />
              </div>

              <div>
                <h3 className="font-bold text-lg mb-1">Knowledge Ingestion</h3>
                <p className="text-slate-500 text-sm mb-5">Inject factual data into the RAG framework.</p>
                <div className="flex bg-slate-100 p-1 rounded-2xl mb-6">
                  <button type="button" onClick={() => setDataType('text')} disabled={isIngesting || !currentSessionId} className={`flex-1 py-3 rounded-xl text-sm font-medium transition ${dataType === 'text' ? 'bg-white shadow' : 'text-slate-600'} disabled:opacity-50 disabled:cursor-not-allowed`}>Raw Text</button>
                  <button type="button" onClick={() => setDataType('file')} disabled={isIngesting || !currentSessionId} className={`flex-1 py-3 rounded-xl text-sm font-medium transition ${dataType === 'file' ? 'bg-white shadow' : 'text-slate-600'} disabled:opacity-50 disabled:cursor-not-allowed`}>Upload Document</button>
                </div>

                <form onSubmit={handleIngestData} className="space-y-5">
                  <div>
                    <label className="block text-sm font-semibold text-slate-600 mb-2">Source Title</label>
                    <input type="text" value={segmentTitle} onChange={(e) => setSegmentTitle(e.target.value)} placeholder={dataType === 'text' ? "e.g., Physics Laws" : "Defaults to filename"} disabled={isIngesting || !currentSessionId} className="w-full px-4 py-3 border border-slate-200 rounded-2xl focus:outline-none focus:border-indigo-300 disabled:bg-slate-100 disabled:cursor-not-allowed" />
                  </div>

                  {dataType === 'text' ? (
                    <div>
                      <label className="block text-sm font-semibold text-slate-600 mb-2">Raw Text Content</label>
                      <textarea rows="5" value={textContent} onChange={(e) => setTextContent(e.target.value)} placeholder="Paste context streams..." disabled={isIngesting || !currentSessionId} className="w-full px-4 py-3 border border-slate-200 rounded-2xl resize-y focus:outline-none focus:border-indigo-300 disabled:bg-slate-100 disabled:cursor-not-allowed" />
                    </div>
                  ) : (
                    <label htmlFor="fileInput" className={`block cursor-pointer ${(isIngesting || !currentSessionId) ? 'pointer-events-none opacity-50' : ''}`}>
                      <div className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all group ${selectedFile ? 'border-emerald-300 bg-emerald-50/30' : 'border-slate-300 hover:border-indigo-400 hover:bg-indigo-50/30'}`}>
                        <div className="flex flex-col items-center gap-3">
                          <div className={`w-14 h-14 rounded-2xl flex items-center justify-center transition-colors shadow-sm ${selectedFile ? 'bg-emerald-100' : 'bg-slate-100 group-hover:bg-indigo-100'}`}>
                            {selectedFile ? (
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                            ) : (
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-7 w-7 text-slate-400 group-hover:text-indigo-600 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" /></svg>
                            )}
                          </div>
                          <div>
                            {selectedFile ? (<><p className="text-sm font-bold text-emerald-700 truncate max-w-[220px]" title={selectedFile.name}>{selectedFile.name}</p><p className="text-xs text-slate-500 mt-1">Click to change selection</p></>) : (<><span className="text-sm font-semibold text-indigo-600 group-hover:text-indigo-700">Click to upload</span><span className="text-sm text-slate-500"> your document</span></>)}
                          </div>
                          <p className={`text-xs px-3 py-1 rounded-full ${selectedFile ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-400'}`}>{selectedFile ? 'File Ready' : '.txt files only'}</p>
                        </div>
                      </div>
                      <input id="fileInput" type="file" accept=".txt" onChange={(e) => setSelectedFile(e.target.files[0])} disabled={isIngesting || !currentSessionId} className="hidden" />
                    </label>
                  )}

                  <button type="submit" disabled={isIngesting || !currentSessionId} className="w-full bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-300 text-white font-semibold py-4 rounded-2xl transition flex items-center justify-center gap-2">
                    {isIngesting ? (<><svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>Processing...</>) : "Add to Knowledge Graph"}
                  </button>
                </form>

                {uploadStatus && (
                  <div className={`mt-4 p-4 rounded-2xl text-sm ${uploadStatus.includes('Success') ? 'bg-emerald-50 text-emerald-800 border border-emerald-200' : 'bg-amber-50 text-amber-800 border border-amber-200'}`}>{uploadStatus}</div>
                )}
              </div>

              <div>
                <h4 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-3">Indexed Sources ({ingestedFiles.length})</h4>
                <div className="space-y-3">
                  {ingestedFiles.length === 0 ? (
                    <p className="text-slate-400 text-sm italic">No active files cataloged.</p>
                  ) : (
                    ingestedFiles.map((file) => (
                      <div key={file.id} className="bg-white border border-slate-200 rounded-2xl p-4 flex items-center justify-between">
                        <div className="truncate pr-2"><div className="font-medium text-sm">📄 {file.file_name}</div></div>
                        <button onClick={() => setActiveViewFile(file)} className="text-xs font-semibold bg-slate-100 hover:bg-slate-200 px-4 py-2 rounded-xl transition">View</button>
                      </div>
                    ))
                  )}
                </div>
              </div>

              <div>
                <h4 className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-3">
                  Knowledge Communities ({communities.length})
                </h4>
                <div className="space-y-4 max-h-[400px] overflow-y-auto pr-2">
                  {communities.length === 0 ? (
                    <p className="text-slate-400 text-sm italic">No communities detected yet.</p>
                  ) : (
                    communities.map((comm) => (
                      <div key={comm.id} className="bg-slate-50 border border-slate-200 rounded-2xl p-4 hover:shadow-md transition">
                        <div className="flex items-center justify-between mb-2">
                          <h5 className="font-bold text-sm text-slate-800">{comm.title}</h5>
                        </div>
                        
                        {/* ✅ FIX 3: Markdown rendering for summary */}
                        <div className="text-sm text-slate-600 mb-3 leading-relaxed prose prose-sm max-w-none">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
                              ul: ({ node, ...props }) => <ul className="list-disc pl-5 mb-2 space-y-1" {...props} />,
                              ol: ({ node, ...props }) => <ol className="list-decimal pl-5 mb-2 space-y-1" {...props} />,
                              li: ({ node, ...props }) => <li className="pl-1" {...props} />,
                              strong: ({ node, ...props }) => <strong className="font-bold text-slate-800" {...props} />,
                              a: ({ node, ...props }) => <a className="text-indigo-600 underline" target="_blank" rel="noopener noreferrer" {...props} />
                            }}
                          >
                            {comm.summary}
                          </ReactMarkdown>
                        </div>

                        <div>
                          <p className="text-xs font-semibold text-slate-500 mb-1.5">Top Entities:</p>
                          <div className="flex flex-wrap gap-1.5">
                            {comm.top_entities?.map((entity, idx) => (
                              <span key={idx} className="text-xs bg-white border border-slate-200 text-slate-700 px-2 py-1 rounded-lg shadow-sm">
                                {entity}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {activeViewFile && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[9999] p-4">
          <div className="bg-white w-full max-w-3xl rounded-3xl shadow-2xl overflow-hidden max-h-[90vh] flex flex-col">
            <div className="px-8 py-5 border-b border-slate-100 flex items-center justify-between bg-slate-50">
              <h3 className="font-bold text-lg">Inspect Context: {activeViewFile.file_name}</h3>
            </div>
            <div className="flex-1 p-8 overflow-y-auto bg-slate-50">
              <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-slate-700">{activeViewFile.text}</pre>
            </div>
            <div className="p-6 border-t border-slate-100 bg-slate-50 flex justify-end">
              <button onClick={() => setActiveViewFile(null)} className="bg-indigo-600 text-white px-8 py-3 rounded-2xl font-semibold hover:bg-indigo-700 transition">Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}