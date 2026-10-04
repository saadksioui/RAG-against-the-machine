import React, { useState } from 'react';

type Message = {
   role: 'user' | 'assistant';
   content: string;
};

const App = () => {
   const [messages, setMessages] = useState<Message[]>([]);
   const [input, setInput] = useState('');
   const [isLoading, setIsLoading] = useState(false);
   const sendMessage = async (e: React.FormEvent) => {
      e.preventDefault();
      if (!input.trim() || isLoading) return;

      const userMessage = input;
      setInput('');
      setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
      setIsLoading(true);

      try {
         const response = await fetch('http://localhost:8000/answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
               query: userMessage,
               k: 5,
               search_mode: 'hybrid'
            }),
         });

         if (!response.ok) throw new Error('API request failed');
         console.log(response);
         

         const data = await response.json();

         setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: data.answer || data.result || "Error: No answer key found in JSON" },
         ]);
      } catch (error) {
         setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: 'Connection error. Is the FastAPI server running?' },
         ]);
      } finally {
         setIsLoading(false);
      }
   };
   return (
      <div className="flex flex-col h-screen bg-gray-900 text-gray-100 font-sans">
         <header className="bg-gray-800 border-b border-gray-700 p-4 text-center shadow-md">
            <h1 className="text-xl font-bold tracking-wider text-green-400">RAG AGAINST THE MACHINE</h1>
         </header>

         <div className="flex-1 overflow-y-auto p-4 space-y-6 max-w-4xl w-full mx-auto">
            {messages.length === 0 && (
               <div className="text-center text-gray-500 mt-20">
                  <p>System initialized. Ready to query the codebase.</p>
               </div>
            )}

            {messages.map((msg, index) => (
               <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] rounded-lg p-4 shadow-sm ${msg.role === 'user'
                        ? 'bg-green-600 text-white rounded-br-none'
                        : 'bg-gray-800 border border-gray-700 text-gray-200 rounded-bl-none'
                     }`}
                  >
                     <p className="whitespace-pre-wrap">{msg.content}</p>
                  </div>
               </div>
            ))}
            {isLoading && (
               <div className="flex justify-start">
                  <div className="bg-gray-800 border border-gray-700 text-gray-400 rounded-lg p-4 rounded-bl-none animate-pulse">
                     Consulting the matrix...
                  </div>
               </div>
            )}
         </div>

         <div className="bg-gray-800 border-t border-gray-700 p-4">
            <form onSubmit={sendMessage} className="max-w-4xl mx-auto flex gap-4">
               <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask a question about the vLLM codebase..."
                  className="flex-1 bg-gray-900 text-white border border-gray-700 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-green-500 transition-all"
                  disabled={isLoading}
               />
               <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="bg-green-600 hover:bg-green-500 text-white font-semibold px-6 py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
               >
                  Send
               </button>
            </form>
         </div>
      </div>
   )
}

export default App