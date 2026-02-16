import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Zap, Layers, Box, Shield } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = import.meta.env.VITE_API_URL;

const LiDARScanner = () => {
  const [data, setData] = useState(null);
  const [isScanning, setIsScanning] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const processFile = async (file) => {
    if (!file) return;
    setIsScanning(true);
    setData(null);
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${BACKEND_URL}/process_lidar`, formData);
      setData(response.data);
    } catch (err) {
      console.error("Backend Error:", err);
      alert("Connection failed. Ensure backend is running.");
    } finally {
      setIsScanning(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true);
    else if (e.type === "dragleave") setDragActive(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Upload Card */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
        className={`relative group rounded-3xl overflow-hidden transition-all duration-500 ${
          dragActive ? 'ring-2 ring-cyan-400' : ''
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {/* Card background with subtle gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-cyan-500/10 via-transparent to-transparent" />
        
        {/* Scan line animation */}
        {isScanning && (
          <motion.div
            className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-cyan-400 to-transparent z-10"
            style={{ boxShadow: '0 0 20px rgba(34, 211, 238, 0.5)' }}
            initial={{ top: 0 }}
            animate={{ top: '100%' }}
            transition={{ duration: 1.8, repeat: Infinity, ease: 'linear' }}
          />
        )}

        {/* Content */}
        <div className="relative z-10 p-10 flex flex-col items-center justify-center min-h-[360px] border border-slate-800 rounded-3xl">
          {/* Status dot */}
          <div className="absolute top-5 left-5 flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${isScanning ? 'bg-cyan-400 animate-pulse' : 'bg-emerald-400'}`} />
            <span className="text-[11px] text-slate-500 font-medium tracking-wide">
              {isScanning ? 'PROCESSING' : 'READY'}
            </span>
          </div>

          {/* Upload icon */}
          <motion.div 
            className="relative mb-6"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400 }}
          >
            <div className="absolute inset-0 bg-cyan-500/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition-opacity" />
            <div className="relative p-5 rounded-2xl bg-slate-800/80 border border-slate-700/50">
              <Upload className={`w-8 h-8 ${isScanning ? 'text-cyan-400' : 'text-slate-400 group-hover:text-cyan-400'} transition-colors`} />
            </div>
          </motion.div>

          <h2 className="text-2xl font-bold text-white mb-2">
            Upload Point Cloud
          </h2>
          <p className="text-slate-500 text-sm mb-8 text-center">
            Drop your .bin LiDAR file here
          </p>

          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => processFile(e.target.files[0])}
            className="hidden"
            accept=".bin"
          />

          <motion.button
            onClick={() => fileInputRef.current.click()}
            disabled={isScanning}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="px-8 py-3.5 bg-gradient-to-r from-cyan-500 to-cyan-400 hover:from-cyan-400 hover:to-cyan-300 disabled:from-slate-700 disabled:to-slate-700 text-slate-900 disabled:text-slate-500 font-semibold rounded-xl transition-all shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30"
          >
            {isScanning ? 'Processing...' : 'Select File'}
          </motion.button>

          <p className="mt-6 text-[11px] text-slate-600 tracking-wide">
            KITTI Format • Binary Stream
          </p>
        </div>
      </motion.div>

      {/* Results Card */}
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="relative rounded-3xl overflow-hidden"
      >
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-900 to-slate-800" />
        
        <div className="relative z-10 p-8 min-h-[360px] border border-slate-800 rounded-3xl">
          <AnimatePresence mode="wait">
            {!data ? (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-full flex flex-col items-center justify-center text-center py-12"
              >
                <div className="p-4 rounded-2xl bg-slate-800/50 mb-4">
                  <Layers className="w-8 h-8 text-slate-600" />
                </div>
                <h3 className="text-lg font-semibold text-slate-500 mb-2">No Data Yet</h3>
                <p className="text-slate-600 text-sm">
                  Upload a file to see results
                </p>
              </motion.div>
            ) : (
              <motion.div
                key="results"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="space-y-6"
              >
                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-3">
                  <Stat icon={<Zap className="w-4 h-4 text-cyan-400" />} label="Latency" value={`${data.processing_latency_ms}ms`} />
                  <Stat icon={<Layers className="w-4 h-4 text-cyan-400" />} label="Raw" value={data.original_points?.toLocaleString()} />
                  <Stat icon={<Box className="w-4 h-4 text-cyan-400" />} label="Processed" value={data.processed_points?.toLocaleString()} />
                  <Stat label="Source" value={data.filename} small />
                </div>

                {/* Main Result */}
                <div className="bg-slate-800/50 rounded-2xl p-6 border border-slate-700/50">
                  <div className="flex items-center justify-between mb-4">
                    <span className="text-xs text-slate-500 font-medium tracking-wide uppercase">Prediction</span>
                    <div className="flex items-center gap-1.5 text-emerald-400">
                      <Shield className="w-3.5 h-3.5" />
                      <span className="text-[11px] font-medium">Verified</span>
                    </div>
                  </div>

                  <div className="flex items-end justify-between mb-6">
                    <h3 className={`text-4xl font-bold ${data.prediction.includes('CAR') ? 'text-emerald-400' : 'text-white'}`}>
                      {data.prediction}
                    </h3>
                    <div className="text-right">
                      <p className="text-[11px] text-slate-500 mb-1">Confidence</p>
                      <p className="text-3xl font-bold text-cyan-400 font-mono">{data.confidence}</p>
                    </div>
                  </div>

                  {/* Confidence Bar */}
                  <div className="h-2 bg-slate-900 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-cyan-500 to-cyan-400 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: data.confidence }}
                      transition={{ duration: 1, ease: 'easeOut' }}
                      style={{ boxShadow: '0 0 12px rgba(34, 211, 238, 0.4)' }}
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
};

const Stat = ({ icon, label, value, small }) => (
  <div className="bg-slate-800/30 rounded-xl p-4 border border-slate-800/50">
    <div className="flex items-center gap-2 mb-1">
      {icon}
      <span className="text-[11px] text-slate-500 font-medium uppercase tracking-wide">{label}</span>
    </div>
    <p className={`font-semibold text-white truncate ${small ? 'text-sm font-mono text-slate-400' : 'text-lg'}`}>
      {value}
    </p>
  </div>
);

export default LiDARScanner;