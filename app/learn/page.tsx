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
                {language === 'en' ? 'Learn Everything You Need To Be An AI Researcher' : '从零开始学习AI'}
              </span>
            </h1>
            <p className="text-xl text-slate-300 mb-8">
              {language === 'en' 
                ? 'Master the fundamentals and publish your own papers'
                : '掌握基础知识，构建你自己的神经网络'}
            </p>
            <div className="max-w-3xl mx-auto bg-amber-500/10 border border-amber-500/30 rounded-xl p-6 mb-8">
              <p className="text-amber-200 text-sm leading-relaxed">
                {language === 'en'
                  ? 'Under active development'
                  : '正在积极开发中'}
              </p>
            </div>
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
                  href="/learn/neural-networks/architecture-of-a-network"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-purple-500/50 hover:shadow-xl hover:shadow-purple-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-purple-400 transition-colors">
                      <span className="mr-2">1.</span>{language === 'en' ? 'Architecture of a Network' : '网络架构'}
                    </h3>
                    <svg className="w-5 h-5 text-purple-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Understanding neural network structure and design'
                      : '理解神经网络结构和设计'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neural-networks/building-a-layer"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-pink-500/50 hover:shadow-xl hover:shadow-pink-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-pink-400 transition-colors">
                      <span className="mr-2">2.</span>{language === 'en' ? 'Building a Layer' : '构建层'}
                    </h3>
                    <svg className="w-5 h-5 text-pink-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Constructing individual network layers'
                      : '构建单个网络层'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neural-networks/implementing-a-network"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-indigo-500/50 hover:shadow-xl hover:shadow-indigo-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-indigo-400 transition-colors">
                      <span className="mr-2">3.</span>{language === 'en' ? 'Implementing a Network' : '实现网络'}
                    </h3>
                    <svg className="w-5 h-5 text-indigo-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Putting together a complete neural network'
                      : '组装完整的神经网络'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neural-networks/the-chain-rule"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-rose-500/50 hover:shadow-xl hover:shadow-rose-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-rose-400 transition-colors">
                      <span className="mr-2">4.</span>{language === 'en' ? 'The Chain Rule' : '链式法则'}
                    </h3>
                    <svg className="w-5 h-5 text-rose-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Mathematical foundation of backpropagation'
                      : '反向传播的数学基础'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neural-networks/calculating-gradients"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-blue-500/50 hover:shadow-xl hover:shadow-blue-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-blue-400 transition-colors">
                      <span className="mr-2">5.</span>{language === 'en' ? 'Calculating Gradients' : '计算梯度'}
                    </h3>
                    <svg className="w-5 h-5 text-blue-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Computing derivatives for network training'
                      : '计算网络训练的导数'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neural-networks/backpropagation-in-action"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-cyan-500/50 hover:shadow-xl hover:shadow-cyan-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 transition-colors">
                      <span className="mr-2">6.</span>{language === 'en' ? 'Backpropagation in Action' : '反向传播实战'}
                    </h3>
                    <svg className="w-5 h-5 text-cyan-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Understanding the backpropagation algorithm'
                      : '理解反向传播算法'}
                  </p>
                </Link>

                <Link 
                  href="/learn/neural-networks/implementing-backpropagation"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-teal-500/50 hover:shadow-xl hover:shadow-teal-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-teal-400 transition-colors">
                      <span className="mr-2">7.</span>{language === 'en' ? 'Implementing Backpropagation' : '实现反向传播'}
                    </h3>
                    <svg className="w-5 h-5 text-teal-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Coding the backpropagation algorithm from scratch'
                      : '从零开始编写反向传播算法'}
                  </p>
                </Link>
              </div>
            </div>

            {/* Attention Mechanism Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-pink-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-white">
                    {language === 'en' ? 'Attention Mechanism' : '注意力机制'}
                  </h2>
                  <p className="text-slate-400">
                    {language === 'en' ? 'Understanding attention and self-attention' : '理解注意力和自注意力'}
                  </p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link 
                  href="/learn/attention-mechanism/what-is-attention"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-red-500/50 hover:shadow-xl hover:shadow-red-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-red-400 transition-colors">
                      <span className="mr-2">1.</span>{language === 'en' ? 'What is Attention' : '什么是注意力'}
                    </h3>
                    <svg className="w-5 h-5 text-red-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Understanding the attention mechanism'
                      : '理解注意力机制'}
                  </p>
                </Link>

                <Link 
                  href="/learn/attention-mechanism/self-attention-from-scratch"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-pink-500/50 hover:shadow-xl hover:shadow-pink-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-pink-400 transition-colors">
                      <span className="mr-2">2.</span>{language === 'en' ? 'Self Attention from Scratch' : '从零开始自注意力'}
                    </h3>
                    <svg className="w-5 h-5 text-pink-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Building self-attention from the ground up'
                      : '从零开始构建自注意力'}
                  </p>
                </Link>

                <Link 
                  href="/learn/attention-mechanism/calculating-attention-scores"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-rose-500/50 hover:shadow-xl hover:shadow-rose-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-rose-400 transition-colors">
                      <span className="mr-2">3.</span>{language === 'en' ? 'Calculating Attention Scores' : '计算注意力分数'}
                    </h3>
                    <svg className="w-5 h-5 text-rose-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Computing query-key-value similarities'
                      : '计算查询-键-值相似度'}
                  </p>
                </Link>

                <Link 
                  href="/learn/attention-mechanism/applying-attention-weights"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-fuchsia-500/50 hover:shadow-xl hover:shadow-fuchsia-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-fuchsia-400 transition-colors">
                      <span className="mr-2">4.</span>{language === 'en' ? 'Applying Attention Weights' : '应用注意力权重'}
                    </h3>
                    <svg className="w-5 h-5 text-fuchsia-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Using attention scores to weight values'
                      : '使用注意力分数加权值'}
                  </p>
                </Link>

                <Link 
                  href="/learn/attention-mechanism/multi-head-attention"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-purple-500/50 hover:shadow-xl hover:shadow-purple-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-purple-400 transition-colors">
                      <span className="mr-2">5.</span>{language === 'en' ? 'Multi Head Attention' : '多头注意力'}
                    </h3>
                    <svg className="w-5 h-5 text-purple-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Parallel attention mechanisms'
                      : '并行注意力机制'}
                  </p>
                </Link>

                <Link 
                  href="/learn/attention-mechanism/attention-in-code"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-violet-500/50 hover:shadow-xl hover:shadow-violet-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-violet-400 transition-colors">
                      <span className="mr-2">6.</span>{language === 'en' ? 'Attention in Code' : '注意力代码实现'}
                    </h3>
                    <svg className="w-5 h-5 text-violet-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Implementing attention mechanisms in Python'
                      : '用Python实现注意力机制'}
                  </p>
                </Link>
              </div>
            </div>

            {/* Transformer Feedforward Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-white">
                    {language === 'en' ? 'Transformer Feedforward' : 'Transformer前馈网络'}
                  </h2>
                  <p className="text-slate-400">
                    {language === 'en' ? 'Feedforward networks and Mixture of Experts' : '前馈网络和专家混合'}
                  </p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link 
                  href="/learn/transformer-feedforward/the-feedforward-layer"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-blue-500/50 hover:shadow-xl hover:shadow-blue-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-blue-400 transition-colors">
                      <span className="mr-2">1.</span>{language === 'en' ? 'The Feedforward Layer' : '前馈层'}
                    </h3>
                    <svg className="w-5 h-5 text-blue-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Understanding transformer feedforward networks'
                      : '理解Transformer前馈网络'}
                  </p>
                </Link>

                <Link 
                  href="/learn/transformer-feedforward/what-is-mixture-of-experts"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-indigo-500/50 hover:shadow-xl hover:shadow-indigo-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-indigo-400 transition-colors">
                      <span className="mr-2">2.</span>{language === 'en' ? 'What is Mixture of Experts' : '什么是专家混合'}
                    </h3>
                    <svg className="w-5 h-5 text-indigo-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Introduction to MoE architecture'
                      : 'MoE架构介绍'}
                  </p>
                </Link>

                <Link 
                  href="/learn/transformer-feedforward/the-expert"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-cyan-500/50 hover:shadow-xl hover:shadow-cyan-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-cyan-400 transition-colors">
                      <span className="mr-2">3.</span>{language === 'en' ? 'The Expert' : '专家'}
                    </h3>
                    <svg className="w-5 h-5 text-cyan-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Understanding individual expert networks'
                      : '理解单个专家网络'}
                  </p>
                </Link>

                <Link 
                  href="/learn/transformer-feedforward/the-gate"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-teal-500/50 hover:shadow-xl hover:shadow-teal-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-teal-400 transition-colors">
                      <span className="mr-2">4.</span>{language === 'en' ? 'The Gate' : '门控'}
                    </h3>
                    <svg className="w-5 h-5 text-teal-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Routing and gating mechanisms in MoE'
                      : 'MoE中的路由和门控机制'}
                  </p>
                </Link>

                <Link 
                  href="/learn/transformer-feedforward/combining-experts"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-sky-500/50 hover:shadow-xl hover:shadow-sky-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-sky-400 transition-colors">
                      <span className="mr-2">5.</span>{language === 'en' ? 'Combining Experts' : '组合专家'}
                    </h3>
                    <svg className="w-5 h-5 text-sky-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Merging multiple expert outputs'
                      : '合并多个专家输出'}
                  </p>
                </Link>

                <Link 
                  href="/learn/transformer-feedforward/moe-in-a-transformer"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-emerald-500/50 hover:shadow-xl hover:shadow-emerald-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-emerald-400 transition-colors">
                      <span className="mr-2">6.</span>{language === 'en' ? 'MoE in a Transformer' : 'Transformer中的MoE'}
                    </h3>
                    <svg className="w-5 h-5 text-emerald-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Integrating mixture of experts in transformers'
                      : '在Transformer中集成专家混合'}
                  </p>
                </Link>

                <Link 
                  href="/learn/transformer-feedforward/moe-in-code"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-green-500/50 hover:shadow-xl hover:shadow-green-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-green-400 transition-colors">
                      <span className="mr-2">7.</span>{language === 'en' ? 'MoE in Code' : 'MoE代码实现'}
                    </h3>
                    <svg className="w-5 h-5 text-green-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Implementing mixture of experts in Python'
                      : '用Python实现专家混合'}
                  </p>
                </Link>

                <Link 
                  href="/learn/transformer-feedforward/the-deepseek-mlp"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-lime-500/50 hover:shadow-xl hover:shadow-lime-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-lime-400 transition-colors">
                      <span className="mr-2">8.</span>{language === 'en' ? 'The DeepSeek MLP' : 'DeepSeek MLP'}
                    </h3>
                    <svg className="w-5 h-5 text-lime-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'DeepSeek\'s advanced MLP architecture'
                      : 'DeepSeek的高级MLP架构'}
                  </p>
                </Link>
              </div>
            </div>

            {/* Building a Transformer Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-white">
                    {language === 'en' ? 'Building a Transformer' : '构建Transformer'}
                  </h2>
                  <p className="text-slate-400">
                    {language === 'en' ? 'Complete transformer implementation from scratch' : '从零开始完整实现Transformer'}
                  </p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link 
                  href="/learn/building-a-transformer/transformer-architecture"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-orange-500/50 hover:shadow-xl hover:shadow-orange-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-orange-400 transition-colors">
                      <span className="mr-2">1.</span>{language === 'en' ? 'Transformer Architecture' : 'Transformer架构'}
                    </h3>
                    <svg className="w-5 h-5 text-orange-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Understanding the complete transformer structure'
                      : '理解完整的Transformer结构'}
                  </p>
                </Link>

                <Link 
                  href="/learn/building-a-transformer/rope-positional-encoding"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-amber-500/50 hover:shadow-xl hover:shadow-amber-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-amber-400 transition-colors">
                      <span className="mr-2">2.</span>{language === 'en' ? 'RoPE Positional Encoding' : 'RoPE位置编码'}
                    </h3>
                    <svg className="w-5 h-5 text-amber-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Rotary position embeddings for transformers'
                      : 'Transformer的旋转位置嵌入'}
                  </p>
                </Link>

                <Link 
                  href="/learn/building-a-transformer/building-a-transformer-block"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-yellow-500/50 hover:shadow-xl hover:shadow-yellow-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-yellow-400 transition-colors">
                      <span className="mr-2">3.</span>{language === 'en' ? 'Building a Transformer Block' : '构建Transformer块'}
                    </h3>
                    <svg className="w-5 h-5 text-yellow-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Constructing individual transformer layers'
                      : '构建单个Transformer层'}
                  </p>
                </Link>

                <Link 
                  href="/learn/building-a-transformer/the-final-linear-layer"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-red-500/50 hover:shadow-xl hover:shadow-red-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-red-400 transition-colors">
                      <span className="mr-2">4.</span>{language === 'en' ? 'The Final Linear Layer' : '最终线性层'}
                    </h3>
                    <svg className="w-5 h-5 text-red-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Output projection and prediction head'
                      : '输出投影和预测头'}
                  </p>
                </Link>

                <Link 
                  href="/learn/building-a-transformer/full-transformer-in-code"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-pink-500/50 hover:shadow-xl hover:shadow-pink-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-pink-400 transition-colors">
                      <span className="mr-2">5.</span>{language === 'en' ? 'Full Transformer in Code' : '完整Transformer代码'}
                    </h3>
                    <svg className="w-5 h-5 text-pink-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Complete transformer implementation'
                      : '完整的Transformer实现'}
                  </p>
                </Link>

                <Link 
                  href="/learn/building-a-transformer/training-a-transformer"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-rose-500/50 hover:shadow-xl hover:shadow-rose-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-rose-400 transition-colors">
                      <span className="mr-2">6.</span>{language === 'en' ? 'Training a Transformer' : '训练Transformer'}
                    </h3>
                    <svg className="w-5 h-5 text-rose-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Training process and optimization'
                      : '训练过程和优化'}
                  </p>
                </Link>
              </div>
            </div>

            {/* Large Language Models Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-white">
                    {language === 'en' ? 'Large Language Models' : '大型语言模型'}
                  </h2>
                  <p className="text-slate-400">
                    {language === 'en' ? 'Understanding LLM training and optimization' : '理解LLM训练和优化'}
                  </p>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link 
                  href="/learn/large-language-models/batch-size-vs-sequence-length"
                  className="group bg-slate-800/50 border border-slate-600/50 rounded-xl p-6 hover:border-emerald-500/50 hover:shadow-xl hover:shadow-emerald-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-white group-hover:text-emerald-400 transition-colors">
                      <span className="mr-2">1.</span>{language === 'en' ? 'Batch Size vs Sequence Length' : '批量大小与序列长度'}
                    </h3>
                    <svg className="w-5 h-5 text-emerald-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-slate-400 text-sm">
                    {language === 'en' 
                      ? 'Understanding the trade-offs between batch size and sequence length'
                      : '理解批量大小和序列长度之间的权衡'}
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

