'use client';

import Link from "next/link";
import { useLanguage } from "@/components/providers/language-provider";

export default function LearnPage() {
  const { language } = useLanguage();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 via-purple-600/10 to-blue-600/10"></div>
        
        <div className="relative container mx-auto px-6">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                {language === 'en' ? 'Learn AI from Scratch' : '从零开始学习AI'}
              </span>
            </h1>
            <p className="text-xl text-slate-300 mb-8">
              {language === 'en' 
                ? 'Master the fundamentals and build your own neural networks'
                : '掌握基础知识，构建你自己的神经网络'}
            </p>
          </div>
        </div>
      </section>

      {/* Course Modules */}
      <section className="py-12">
        <div className="container mx-auto px-6">
          <div className="max-w-6xl mx-auto space-y-12">
            
            {/* Math Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-white">
                    {language === 'en' ? 'Mathematics Fundamentals' : '数学基础'}
                  </h2>
                  <p className="text-slate-400">
                    {language === 'en' ? 'Essential math concepts for AI' : 'AI必备的数学概念'}
                  </p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link 
                  href="/learn/math/functions"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-cyan-500/50 hover:shadow-xl hover:shadow-cyan-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 transition-colors">
                      <span className="mr-2">1.</span>{language === 'en' ? 'Functions' : '函数'}
                    </h3>
                    <svg className="w-5 h-5 text-cyan-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Linear, quadratic, and activation functions'
                      : '线性、二次和激活函数'}
                  </p>
                </Link>

                <Link 
                  href="/learn/math/derivatives"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-blue-500/50 hover:shadow-xl hover:shadow-blue-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-blue-400 transition-colors">
                      <span className="mr-2">2.</span>{language === 'en' ? 'Derivatives' : '导数'}
                    </h3>
                    <svg className="w-5 h-5 text-blue-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Understanding rates of change and gradients'
                      : '理解变化率和梯度'}
                  </p>
                </Link>

                <Link 
                  href="/learn/math/vectors"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-violet-500/50 hover:shadow-xl hover:shadow-violet-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-violet-400 transition-colors">
                      <span className="mr-2">3.</span>{language === 'en' ? 'Vectors' : '向量'}
                    </h3>
                    <svg className="w-5 h-5 text-violet-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Understanding magnitude, direction, and vector operations'
                      : '理解大小、方向和向量运算'}
                  </p>
                </Link>

                <Link 
                  href="/learn/math/matrices"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-teal-500/50 hover:shadow-xl hover:shadow-teal-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-teal-400 transition-colors">
                      <span className="mr-2">4.</span>{language === 'en' ? 'Matrices' : '矩阵'}
                    </h3>
                    <svg className="w-5 h-5 text-teal-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Matrix operations and transformations'
                      : '矩阵运算和变换'}
                  </p>
                </Link>

                <Link 
                  href="/learn/math/gradients"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-orange-500/50 hover:shadow-xl hover:shadow-orange-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-orange-400 transition-colors">
                      <span className="mr-2">5.</span>{language === 'en' ? 'Gradients' : '梯度'}
                    </h3>
                    <svg className="w-5 h-5 text-orange-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Partial derivatives and gradient descent'
                      : '偏导数和梯度下降'}
                  </p>
                </Link>
              </div>
            </div>

            {/* Neural Networks Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-white">
                    {language === 'en' ? 'Neural Networks from Scratch' : '从零开始的神经网络'}
                  </h2>
                  <p className="text-slate-400">
                    {language === 'en' ? 'Build neural networks from the ground up' : '从头构建神经网络'}
                  </p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link 
                  href="/learn/neural-networks/introduction"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-purple-500/50 hover:shadow-xl hover:shadow-purple-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-purple-400 transition-colors">
                      <span className="mr-2">6.</span>{language === 'en' ? 'Introduction' : '简介'}
                    </h3>
                    <svg className="w-5 h-5 text-purple-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'What are neural networks and how do they work?'
                      : '什么是神经网络以及它们如何工作？'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neural-networks/forward-propagation"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-pink-500/50 hover:shadow-xl hover:shadow-pink-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-pink-400 transition-colors">
                      <span className="mr-2">7.</span>{language === 'en' ? 'Forward Propagation' : '前向传播'}
                    </h3>
                    <svg className="w-5 h-5 text-pink-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Computing outputs from inputs through the network'
                      : '通过网络从输入计算输出'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neural-networks/backpropagation"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-indigo-500/50 hover:shadow-xl hover:shadow-indigo-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-indigo-400 transition-colors">
                      <span className="mr-2">8.</span>{language === 'en' ? 'Backpropagation' : '反向传播'}
                    </h3>
                    <svg className="w-5 h-5 text-indigo-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'The algorithm that enables learning in neural networks'
                      : '使神经网络能够学习的算法'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neural-networks/training"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-emerald-500/50 hover:shadow-xl hover:shadow-emerald-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-emerald-400 transition-colors">
                      <span className="mr-2">9.</span>{language === 'en' ? 'Training & Optimization' : '训练与优化'}
                    </h3>
                    <svg className="w-5 h-5 text-emerald-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Gradient descent and optimization techniques'
                      : '梯度下降和优化技术'}
                  </p>
                </Link>
              </div>
            </div>

          </div>
        </div>
      </section>
    </div>
  );
}

