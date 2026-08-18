import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import ChatApp from './components/ChatApp';
import Auth from './components/Auth';

function App() {
  const [token, setToken] = useState(localStorage.getItem('token'));

  const handleLogin = (newToken) => {
    localStorage.setItem('token', newToken);
    setToken(newToken);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setToken(null);
  };

  return (
    <Router>
      <Routes>
        <Route 
          path="/auth" 
          element={!token ? <Auth onLoginSuccess={handleLogin} /> : <Navigate to="/" replace />} 
        />
        <Route 
          path="/" 
          element={token ? <ChatApp onLogout={handleLogout} /> : <Navigate to="/auth" replace />} 
        />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

export default App;