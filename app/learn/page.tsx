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

            {/* PyTorch Fundamentals Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-white">
                    {language === 'en' ? 'PyTorch Fundamentals' : 'PyTorch基础'}
                  </h2>
                  <p className="text-slate-400">
                    {language === 'en' ? 'Working with tensors and PyTorch basics' : '使用张量和PyTorch基础'}
                  </p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link 
                  href="/learn/tensors/creating-tensors"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-green-500/50 hover:shadow-xl hover:shadow-green-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-green-400 transition-colors">
                      <span className="mr-2">1.</span>{language === 'en' ? 'Creating Tensors' : '创建张量'}
                    </h3>
                    <svg className="w-5 h-5 text-green-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Building blocks of deep learning'
                      : '深度学习的基本构建块'}
                  </p>
                </Link>

                <Link 
                  href="/learn/tensors/tensor-addition"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-lime-500/50 hover:shadow-xl hover:shadow-lime-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-lime-400 transition-colors">
                      <span className="mr-2">2.</span>{language === 'en' ? 'Tensor Addition' : '张量加法'}
                    </h3>
                    <svg className="w-5 h-5 text-lime-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Element-wise operations on tensors'
                      : '张量的逐元素运算'}
                  </p>
                </Link>

                <Link 
                  href="/learn/tensors/matrix-multiplication"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-emerald-500/50 hover:shadow-xl hover:shadow-emerald-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-emerald-400 transition-colors">
                      <span className="mr-2">3.</span>{language === 'en' ? 'Matrix Multiplication' : '矩阵乘法'}
                    </h3>
                    <svg className="w-5 h-5 text-emerald-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'The core operation in neural networks'
                      : '神经网络中的核心运算'}
                  </p>
                </Link>

                <Link 
                  href="/learn/tensors/transposing-tensors"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-teal-500/50 hover:shadow-xl hover:shadow-teal-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-teal-400 transition-colors">
                      <span className="mr-2">4.</span>{language === 'en' ? 'Transposing Tensors' : '张量转置'}
                    </h3>
                    <svg className="w-5 h-5 text-teal-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Flipping dimensions and axes'
                      : '翻转维度和轴'}
                  </p>
                </Link>

                <Link 
                  href="/learn/tensors/reshaping-tensors"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-cyan-500/50 hover:shadow-xl hover:shadow-cyan-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 transition-colors">
                      <span className="mr-2">5.</span>{language === 'en' ? 'Reshaping Tensors' : '张量重塑'}
                    </h3>
                    <svg className="w-5 h-5 text-cyan-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Changing tensor dimensions'
                      : '改变张量维度'}
                  </p>
                </Link>

                <Link 
                  href="/learn/tensors/indexing-and-slicing"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-sky-500/50 hover:shadow-xl hover:shadow-sky-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-sky-400 transition-colors">
                      <span className="mr-2">6.</span>{language === 'en' ? 'Indexing and Slicing' : '索引和切片'}
                    </h3>
                    <svg className="w-5 h-5 text-sky-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Accessing and extracting tensor elements'
                      : '访问和提取张量元素'}
                  </p>
                </Link>

                <Link 
                  href="/learn/tensors/concatenating-tensors"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-indigo-500/50 hover:shadow-xl hover:shadow-indigo-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-indigo-400 transition-colors">
                      <span className="mr-2">7.</span>{language === 'en' ? 'Concatenating Tensors' : '张量拼接'}
                    </h3>
                    <svg className="w-5 h-5 text-indigo-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Combining multiple tensors'
                      : '组合多个张量'}
                  </p>
                </Link>

                <Link 
                  href="/learn/tensors/creating-special-tensors"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-violet-500/50 hover:shadow-xl hover:shadow-violet-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-violet-400 transition-colors">
                      <span className="mr-2">8.</span>{language === 'en' ? 'Creating Special Tensors' : '创建特殊张量'}
                    </h3>
                    <svg className="w-5 h-5 text-violet-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Zeros, ones, identity matrices and more'
                      : '零张量、单位张量、单位矩阵等'}
                  </p>
                </Link>
              </div>
            </div>

            {/* Neuron From Scratch Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-white">
                    {language === 'en' ? 'Neuron From Scratch' : '从零开始构建神经元'}
                  </h2>
                  <p className="text-slate-400">
                    {language === 'en' ? 'Understanding the fundamental unit of neural networks' : '理解神经网络的基本单元'}
                  </p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link 
                  href="/learn/neuron-from-scratch/what-is-a-neuron"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-yellow-500/50 hover:shadow-xl hover:shadow-yellow-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-yellow-400 transition-colors">
                      <span className="mr-2">1.</span>{language === 'en' ? 'What is a Neuron' : '什么是神经元'}
                    </h3>
                    <svg className="w-5 h-5 text-yellow-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'The basic building block of neural networks'
                      : '神经网络的基本构建块'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neuron-from-scratch/the-linear-step"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-amber-500/50 hover:shadow-xl hover:shadow-amber-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-amber-400 transition-colors">
                      <span className="mr-2">2.</span>{language === 'en' ? 'The Linear Step' : '线性步骤'}
                    </h3>
                    <svg className="w-5 h-5 text-amber-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Weighted sums and bias in neurons'
                      : '神经元中的加权和和偏置'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neuron-from-scratch/the-activation-function"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-orange-500/50 hover:shadow-xl hover:shadow-orange-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-orange-400 transition-colors">
                      <span className="mr-2">3.</span>{language === 'en' ? 'The Activation Function' : '激活函数'}
                    </h3>
                    <svg className="w-5 h-5 text-orange-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Introducing non-linearity to neurons'
                      : '为神经元引入非线性'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neuron-from-scratch/building-a-neuron-in-python"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-red-500/50 hover:shadow-xl hover:shadow-red-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-red-400 transition-colors">
                      <span className="mr-2">4.</span>{language === 'en' ? 'Building a Neuron in Python' : '用Python构建神经元'}
                    </h3>
                    <svg className="w-5 h-5 text-red-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Implementing a single neuron from scratch'
                      : '从零开始实现单个神经元'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neuron-from-scratch/making-a-prediction"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-pink-500/50 hover:shadow-xl hover:shadow-pink-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-pink-400 transition-colors">
                      <span className="mr-2">5.</span>{language === 'en' ? 'Making a Prediction' : '进行预测'}
                    </h3>
                    <svg className="w-5 h-5 text-pink-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'How a neuron processes input to output'
                      : '神经元如何处理输入到输出'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neuron-from-scratch/the-concept-of-loss"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-rose-500/50 hover:shadow-xl hover:shadow-rose-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-rose-400 transition-colors">
                      <span className="mr-2">6.</span>{language === 'en' ? 'The Concept of Loss' : '损失概念'}
                    </h3>
                    <svg className="w-5 h-5 text-rose-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Measuring prediction error'
                      : '测量预测误差'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neuron-from-scratch/the-concept-of-learning"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-fuchsia-500/50 hover:shadow-xl hover:shadow-fuchsia-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-fuchsia-400 transition-colors">
                      <span className="mr-2">7.</span>{language === 'en' ? 'The Concept of Learning' : '学习概念'}
                    </h3>
                    <svg className="w-5 h-5 text-fuchsia-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'How neurons adjust their parameters'
                      : '神经元如何调整其参数'}
                  </p>
                </Link>
              </div>
            </div>

            {/* Activation Functions Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-white">
                    {language === 'en' ? 'Activation Functions' : '激活函数'}
                  </h2>
                  <p className="text-slate-400">
                    {language === 'en' ? 'Understanding different activation functions' : '理解不同的激活函数'}
                  </p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link 
                  href="/learn/activation-functions/relu"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-indigo-500/50 hover:shadow-xl hover:shadow-indigo-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-indigo-400 transition-colors">
                      <span className="mr-2">1.</span>{language === 'en' ? 'ReLU' : 'ReLU'}
                    </h3>
                    <svg className="w-5 h-5 text-indigo-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Rectified Linear Unit - The most popular activation function'
                      : '修正线性单元 - 最流行的激活函数'}
                  </p>
                </Link>

                <Link 
                  href="/learn/activation-functions/sigmoid"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-purple-500/50 hover:shadow-xl hover:shadow-purple-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-purple-400 transition-colors">
                      <span className="mr-2">2.</span>{language === 'en' ? 'Sigmoid' : 'Sigmoid'}
                    </h3>
                    <svg className="w-5 h-5 text-purple-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'The classic S-shaped activation function'
                      : '经典的S形激活函数'}
                  </p>
                </Link>

                <Link 
                  href="/learn/activation-functions/tanh"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-violet-500/50 hover:shadow-xl hover:shadow-violet-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-violet-400 transition-colors">
                      <span className="mr-2">3.</span>{language === 'en' ? 'Tanh' : 'Tanh'}
                    </h3>
                    <svg className="w-5 h-5 text-violet-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Hyperbolic tangent - Zero-centered activation'
                      : '双曲正切 - 零中心激活'}
                  </p>
                </Link>

                <Link 
                  href="/learn/activation-functions/silu"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-blue-500/50 hover:shadow-xl hover:shadow-blue-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-blue-400 transition-colors">
                      <span className="mr-2">4.</span>{language === 'en' ? 'SiLU' : 'SiLU'}
                    </h3>
                    <svg className="w-5 h-5 text-blue-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Sigmoid Linear Unit - The Swish activation'
                      : 'Sigmoid线性单元 - Swish激活'}
                  </p>
                </Link>

                <Link 
                  href="/learn/activation-functions/swiglu"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-cyan-500/50 hover:shadow-xl hover:shadow-cyan-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 transition-colors">
                      <span className="mr-2">5.</span>{language === 'en' ? 'SwiGLU' : 'SwiGLU'}
                    </h3>
                    <svg className="w-5 h-5 text-cyan-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Swish-Gated Linear Unit - Advanced activation'
                      : 'Swish门控线性单元 - 高级激活'}
                  </p>
                </Link>

                <Link 
                  href="/learn/activation-functions/softmax"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-teal-500/50 hover:shadow-xl hover:shadow-teal-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-teal-400 transition-colors">
                      <span className="mr-2">6.</span>{language === 'en' ? 'Softmax' : 'Softmax'}
                    </h3>
                    <svg className="w-5 h-5 text-teal-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Multi-class classification activation function'
                      : '多类分类激活函数'}
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
                      <span className="mr-2">1.</span>{language === 'en' ? 'Introduction' : '简介'}
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
                      <span className="mr-2">2.</span>{language === 'en' ? 'Forward Propagation' : '前向传播'}
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
                      <span className="mr-2">3.</span>{language === 'en' ? 'Backpropagation' : '反向传播'}
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
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-rose-500/50 hover:shadow-xl hover:shadow-rose-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-rose-400 transition-colors">
                      <span className="mr-2">4.</span>{language === 'en' ? 'Training & Optimization' : '训练与优化'}
                    </h3>
                    <svg className="w-5 h-5 text-rose-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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

