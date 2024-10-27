import React, { useState } from 'react';
import { Mosaic } from "react-loading-indicators";
function App() {
  const [sentence, setSentence] = useState('');
  const [targetWord, setTargetWord] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setResponse(null);
    setLoading(true);

    // Create the request payload
    const payload = {
      sentence: sentence,
      target_word: targetWord
    };

    try {
      // Send POST request to the Flask API
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await res.json();

      setTimeout(() =>{
        setResponse(data.sense);
      setLoading(false);
       }, 3000);
    } catch (error) {
      console.error('Error:', error);
      setResponse("Error occurred while fetching sense.");
    }
  };

  return (
    <div className="w-full min-h-screen text-black font-mono bg-[#89ABE3] flex flex-col items-center justify-end">
      {loading && <span  className=" mb-20" > <Mosaic size="small" color={["#EA738D", "#CADCFC", "#fff200"]} /></span>} {response && (
        <div className='w-7/12 mb-20 flex flex-col items-start justify-center bg-[#EA738D] p-5 rounded-lg'>
          <h2 className='pb-2 underline underline-offset-3'>Model Prediction:</h2>
          <p >{response}</p>
        </div>
      )}
      <form onSubmit={handleSubmit} className='flex flex-col gap-3 py-5 items-center  w-full'>
        <div className='w-full flex items-center justify-center'>
          <input
            type="text"
            value={sentence}
            onChange={(e) => setSentence(e.target.value)}
            required
            placeholder='Enter a sentence'
              className="form-control text-lg sm:w-7/12 h-14 rounded-xl border-gray-950 border-4 outline-none shadow-[2px_4px_0px_0px_rgba(0,0,0)] px-4  py-2 "
          />
        </div>
        <div className='flex w-7/12  items-center justify-between '>
          <input
            type="text"
            value={targetWord}
            onChange={(e) => setTargetWord(e.target.value)}
            required
            placeholder='Enter the target word'
              className="form-control text-lg sm:w-8/12 h-14 rounded-xl border-gray-950 border-4 outline-none shadow-[2px_4px_0px_0px_rgba(0,0,0)] px-4  py-2 "
          />
        <button type="submit" o
         className="bg-[#fff200]  hover:bg-[#fff200]/80  border-gray-950 border-4 text-black shadow-[2px_3px_0px_0px_rgba(0,0,0)] text-lg font-medium py-2 px-5 rounded-md">
          Predict</button>
        </div>
      </form>

      
    </div>
  );
}

export default App;
