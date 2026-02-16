import React from 'react';
import { motion } from 'framer-motion';
import { Radar, Github } from 'lucide-react';
import LiDARScanner from './components/LiDARScanner';

function App() {
  return (
    <div className="min-h-screen bg-[#0a0f1a] text-slate-200 overflow-hidden">
      {/* Ambient background */}
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute top-[-20%] left-[-10%] w-[500px] h-[500px] bg-cyan-500/8 rounded-full blur-[100px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-cyan-500/5 rounded-full blur-[120px]" />
        
        {/* Subtle grid */}
        <div 
          className="absolute inset-0 opacity-[0.02]"
          style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, rgba(148, 163, 184, 0.5) 1px, transparent 0)`,
            backgroundSize: '32px 32px',
          }}
        />
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Navigation */}
        <motion.nav 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-16"
        >
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-cyan-500/10 border border-cyan-500/20">
              <Radar className="w-5 h-5 text-cyan-400" />
            </div>
            <span className="text-lg font-bold text-white">TensorStream</span>
          </div>
          
          <a 
            href="#" 
            className="p-2.5 rounded-xl bg-slate-800/50 border border-slate-700/50 text-slate-400 hover:text-white hover:border-slate-600 transition-all"
          >
            <Github className="w-5 h-5" />
          </a>
        </motion.nav>

        {/* Header */}
        <motion.header 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-center mb-14"
        >
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-800/50 border border-slate-700/50 mb-6"
          >
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-400" />
            </span>
            <span className="text-sm text-slate-400">System Online</span>
          </motion.div>

          <h1 className="text-5xl md:text-6xl font-bold text-white mb-4 tracking-tight">
            LiDAR <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-cyan-300">Classification</span>
          </h1>
          
          <p className="text-slate-400 text-lg max-w-lg mx-auto">
            Real-time point cloud analysis powered by PointNet neural networks
          </p>
        </motion.header>

        {/* Main Component */}
        <motion.main 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-20"
        >
          <LiDARScanner />
        </motion.main>

        {/* Tech Stack */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="flex flex-wrap justify-center gap-4 mb-16"
        >
          {['C++ Backend', 'PointNet-3D', 'FastAPI', 'Voxelization'].map((tech, i) => (
            <motion.span
              key={tech}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + i * 0.1 }}
              className="px-4 py-2 rounded-xl bg-slate-800/30 border border-slate-800 text-sm text-slate-500 hover:text-slate-300 hover:border-slate-700 transition-colors cursor-default"
            >
              {tech}
            </motion.span>
          ))}
        </motion.div>

        {/* Footer */}
        <motion.footer 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="text-center pb-8"
        >
          <div className="h-px bg-gradient-to-r from-transparent via-slate-800 to-transparent mb-8" />
          <p className="text-slate-600 text-sm">
            Architected by <span className="text-slate-500">Kashyap Hegde Kota</span> • Arizona State University
          </p>
        </motion.footer>
      </div>
    </div>
  );
}

export default App;