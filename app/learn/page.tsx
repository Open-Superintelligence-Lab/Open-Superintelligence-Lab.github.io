'use client';

import Link from "next/link";

export default function LearnPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gradient-start via-gradient-mid to-gradient-end">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20">
        <div className="absolute inset-0 bg-gradient-to-r from-gradient-accent-1/10 via-gradient-accent-2/10 to-gradient-accent-1/10"></div>

        <div className="relative container mx-auto px-6">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-5xl md:text-6xl font-bold mb-6">
              <span className="bg-gradient-to-r from-gradient-accent-1 via-gradient-accent-2 to-gradient-accent-3 bg-clip-text text-transparent">
                Learn Everything You Need To Be An AI Researcher
              </span>
            </h1>
            <p className="text-xl text-muted-foreground mb-8">
              Master the fundamentals and publish your own papers
            </p>
            <div className="max-w-3xl mx-auto bg-amber-500/10 border border-amber-500/30 rounded-xl p-6 mb-8">
              <p className="text-amber-200 text-sm leading-relaxed">
                Under active development
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
            <div className="bg-gradient-to-br from-card/50 to-background/50 backdrop-blur-sm border border-border/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-gradient-accent-1 to-gradient-accent-3 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-foreground">
                    Mathematics Fundamentals
                  </h2>
                  <p className="text-muted-foreground">
                    Essential math concepts for AI
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link
                  href="/learn/math/functions"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-gradient-accent-3/50 hover:shadow-xl hover:shadow-gradient-accent-3/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-3 transition-colors">
                      <span className="mr-2">1.</span>Functions
                    </h3>
                    <svg className="w-5 h-5 text-gradient-accent-3 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Linear, quadratic, and activation functions
                  </p>
                </Link>

                <Link
                  href="/learn/math/derivatives"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-gradient-accent-1/50 hover:shadow-xl hover:shadow-gradient-accent-1/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-blue-400 transition-colors">
                      <span className="mr-2">2.</span>Derivatives
                    </h3>
                    <svg className="w-5 h-5 text-gradient-accent-1 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Understanding rates of change and gradients
                  </p>
                </Link>

                <Link
                  href="/learn/math/vectors"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-violet-500/50 hover:shadow-xl hover:shadow-violet-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-violet-400 transition-colors">
                      <span className="mr-2">3.</span>Vectors
                    </h3>
                    <svg className="w-5 h-5 text-violet-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Understanding magnitude, direction, and vector operations
                  </p>
                </Link>

                <Link
                  href="/learn/math/matrices"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-teal-500/50 hover:shadow-xl hover:shadow-teal-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-teal-400 transition-colors">
                      <span className="mr-2">4.</span>Matrices
                    </h3>
                    <svg className="w-5 h-5 text-teal-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Matrix operations and transformations
                  </p>
                </Link>

                <Link
                  href="/learn/math/gradients"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-orange-500/50 hover:shadow-xl hover:shadow-orange-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-orange-400 transition-colors">
                      <span className="mr-2">5.</span>Gradients
                    </h3>
                    <svg className="w-5 h-5 text-orange-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Partial derivatives and gradient descent
                  </p>
                </Link>
              </div>
            </div>

            {/* PyTorch Fundamentals Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-foreground">
                    PyTorch Fundamentals
                  </h2>
                  <p className="text-muted-foreground">
                    Working with tensors and PyTorch basics
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link
                  href="/learn/tensors/creating-tensors"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-green-500/50 hover:shadow-xl hover:shadow-green-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-green-400 transition-colors">
                      <span className="mr-2">1.</span>Creating Tensors
                    </h3>
                    <svg className="w-5 h-5 text-green-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Building blocks of deep learning
                  </p>
                </Link>

                <Link
                  href="/learn/tensors/tensor-addition"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-lime-500/50 hover:shadow-xl hover:shadow-lime-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-lime-400 transition-colors">
                      <span className="mr-2">2.</span>Tensor Addition
                    </h3>
                    <svg className="w-5 h-5 text-lime-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Element-wise operations on tensors
                  </p>
                </Link>

                <Link
                  href="/learn/tensors/matrix-multiplication"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-emerald-500/50 hover:shadow-xl hover:shadow-emerald-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-emerald-400 transition-colors">
                      <span className="mr-2">3.</span>Matrix Multiplication
                    </h3>
                    <svg className="w-5 h-5 text-emerald-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    The core operation in neural networks
                  </p>
                </Link>

                <Link
                  href="/learn/tensors/transposing-tensors"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-teal-500/50 hover:shadow-xl hover:shadow-teal-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-teal-400 transition-colors">
                      <span className="mr-2">4.</span>Transposing Tensors
                    </h3>
                    <svg className="w-5 h-5 text-teal-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Flipping dimensions and axes
                  </p>
                </Link>

                <Link
                  href="/learn/tensors/reshaping-tensors"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-gradient-accent-3/50 hover:shadow-xl hover:shadow-gradient-accent-3/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-cyan-400 transition-colors">
                      <span className="mr-2">5.</span>Reshaping Tensors
                    </h3>
                    <svg className="w-5 h-5 text-gradient-accent-3 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Changing tensor dimensions
                  </p>
                </Link>

                <Link
                  href="/learn/tensors/indexing-and-slicing"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-sky-500/50 hover:shadow-xl hover:shadow-sky-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-sky-400 transition-colors">
                      <span className="mr-2">6.</span>Indexing and Slicing
                    </h3>
                    <svg className="w-5 h-5 text-sky-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Accessing and extracting tensor elements
                  </p>
                </Link>

                <Link
                  href="/learn/tensors/concatenating-tensors"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-indigo-500/50 hover:shadow-xl hover:shadow-indigo-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-indigo-400 transition-colors">
                      <span className="mr-2">7.</span>Concatenating Tensors
                    </h3>
                    <svg className="w-5 h-5 text-indigo-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Combining multiple tensors
                  </p>
                </Link>

                <Link
                  href="/learn/tensors/creating-special-tensors"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-violet-500/50 hover:shadow-xl hover:shadow-violet-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-violet-400 transition-colors">
                      <span className="mr-2">8.</span>Creating Special Tensors
                    </h3>
                    <svg className="w-5 h-5 text-violet-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Zeros, ones, identity matrices and more
                  </p>
                </Link>
              </div>
            </div>

            {/* Neuron From Scratch Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-yellow-500 to-orange-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-foreground">
                    Neuron From Scratch
                  </h2>
                  <p className="text-muted-foreground">
                    Understanding the fundamental unit of neural networks
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link
                  href="/learn/neuron-from-scratch/what-is-a-neuron"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-yellow-500/50 hover:shadow-xl hover:shadow-yellow-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-yellow-400 transition-colors">
                      <span className="mr-2">1.</span>What is a Neuron
                    </h3>
                    <svg className="w-5 h-5 text-yellow-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    The basic building block of neural networks
                  </p>
                </Link>

                <Link
                  href="/learn/neuron-from-scratch/the-linear-step"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-amber-500/50 hover:shadow-xl hover:shadow-amber-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-amber-400 transition-colors">
                      <span className="mr-2">2.</span>The Linear Step
                    </h3>
                    <svg className="w-5 h-5 text-amber-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Weighted sums and bias in neurons
                  </p>
                </Link>

                <Link
                  href="/learn/neuron-from-scratch/the-activation-function"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-orange-500/50 hover:shadow-xl hover:shadow-orange-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-orange-400 transition-colors">
                      <span className="mr-2">3.</span>The Activation Function
                    </h3>
                    <svg className="w-5 h-5 text-orange-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Introducing non-linearity to neurons
                  </p>
                </Link>

                <Link
                  href="/learn/neuron-from-scratch/building-a-neuron-in-python"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-red-500/50 hover:shadow-xl hover:shadow-red-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-red-400 transition-colors">
                      <span className="mr-2">4.</span>Building a Neuron in Python
                    </h3>
                    <svg className="w-5 h-5 text-red-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Implementing a single neuron from scratch
                  </p>
                </Link>

                <Link
                  href="/learn/neuron-from-scratch/making-a-prediction"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-pink-500/50 hover:shadow-xl hover:shadow-pink-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-pink-400 transition-colors">
                      <span className="mr-2">5.</span>Making a Prediction
                    </h3>
                    <svg className="w-5 h-5 text-pink-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    How a neuron processes input to output
                  </p>
                </Link>

                <Link
                  href="/learn/neuron-from-scratch/the-concept-of-loss"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-rose-500/50 hover:shadow-xl hover:shadow-rose-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-rose-400 transition-colors">
                      <span className="mr-2">6.</span>The Concept of Loss
                    </h3>
                    <svg className="w-5 h-5 text-rose-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Measuring prediction error
                  </p>
                </Link>

                <Link
                  href="/learn/neuron-from-scratch/the-concept-of-learning"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-fuchsia-500/50 hover:shadow-xl hover:shadow-fuchsia-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-fuchsia-400 transition-colors">
                      <span className="mr-2">7.</span>The Concept of Learning
                    </h3>
                    <svg className="w-5 h-5 text-fuchsia-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    How neurons adjust their parameters
                  </p>
                </Link>
              </div>
            </div>

            {/* Activation Functions Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-foreground">
                    Activation Functions
                  </h2>
                  <p className="text-muted-foreground">
                    Understanding different activation functions
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link
                  href="/learn/activation-functions/relu"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-indigo-500/50 hover:shadow-xl hover:shadow-indigo-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-indigo-400 transition-colors">
                      <span className="mr-2">1.</span>ReLU
                    </h3>
                    <svg className="w-5 h-5 text-indigo-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Rectified Linear Unit - The most popular activation function
                  </p>
                </Link>

                <Link
                  href="/learn/activation-functions/sigmoid"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-gradient-accent-2/50 hover:shadow-xl hover:shadow-gradient-accent-2/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-purple-400 transition-colors">
                      <span className="mr-2">2.</span>Sigmoid
                    </h3>
                    <svg className="w-5 h-5 text-gradient-accent-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    The classic S-shaped activation function
                  </p>
                </Link>

                <Link
                  href="/learn/activation-functions/tanh"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-violet-500/50 hover:shadow-xl hover:shadow-violet-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-violet-400 transition-colors">
                      <span className="mr-2">3.</span>Tanh
                    </h3>
                    <svg className="w-5 h-5 text-violet-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Hyperbolic tangent - Zero-centered activation
                  </p>
                </Link>

                <Link
                  href="/learn/activation-functions/silu"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-gradient-accent-1/50 hover:shadow-xl hover:shadow-gradient-accent-1/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-blue-400 transition-colors">
                      <span className="mr-2">4.</span>SiLU
                    </h3>
                    <svg className="w-5 h-5 text-gradient-accent-1 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Sigmoid Linear Unit - The Swish activation
                  </p>
                </Link>

                <Link
                  href="/learn/activation-functions/swiglu"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-gradient-accent-3/50 hover:shadow-xl hover:shadow-gradient-accent-3/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-cyan-400 transition-colors">
                      <span className="mr-2">5.</span>SwiGLU
                    </h3>
                    <svg className="w-5 h-5 text-gradient-accent-3 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Swish-Gated Linear Unit - Advanced activation
                  </p>
                </Link>

                <Link
                  href="/learn/activation-functions/softmax"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-teal-500/50 hover:shadow-xl hover:shadow-teal-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-teal-400 transition-colors">
                      <span className="mr-2">6.</span>Softmax
                    </h3>
                    <svg className="w-5 h-5 text-teal-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Multi-class classification activation function
                  </p>
                </Link>
              </div>
            </div>

            {/* Neural Networks Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-foreground">
                    Neural Networks from Scratch
                  </h2>
                  <p className="text-muted-foreground">
                    Build neural networks from the ground up
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link
                  href="/learn/neural-networks/architecture-of-a-network"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-gradient-accent-2/50 hover:shadow-xl hover:shadow-gradient-accent-2/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-purple-400 transition-colors">
                      <span className="mr-2">1.</span>Architecture of a Network
                    </h3>
                    <svg className="w-5 h-5 text-gradient-accent-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Understanding neural network structure and design
                  </p>
                </Link>

                <Link
                  href="/learn/neural-networks/building-a-layer"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-pink-500/50 hover:shadow-xl hover:shadow-pink-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-pink-400 transition-colors">
                      <span className="mr-2">2.</span>Building a Layer
                    </h3>
                    <svg className="w-5 h-5 text-pink-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Constructing individual network layers
                  </p>
                </Link>

                <Link
                  href="/learn/neural-networks/implementing-a-network"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-indigo-500/50 hover:shadow-xl hover:shadow-indigo-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-indigo-400 transition-colors">
                      <span className="mr-2">3.</span>Implementing a Network
                    </h3>
                    <svg className="w-5 h-5 text-indigo-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Putting together a complete neural network
                  </p>
                </Link>

                <Link
                  href="/learn/neural-networks/the-chain-rule"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-rose-500/50 hover:shadow-xl hover:shadow-rose-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-rose-400 transition-colors">
                      <span className="mr-2">4.</span>The Chain Rule
                    </h3>
                    <svg className="w-5 h-5 text-rose-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Mathematical foundation of backpropagation
                  </p>
                </Link>

                <Link
                  href="/learn/neural-networks/calculating-gradients"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-gradient-accent-1/50 hover:shadow-xl hover:shadow-gradient-accent-1/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-blue-400 transition-colors">
                      <span className="mr-2">5.</span>Calculating Gradients
                    </h3>
                    <svg className="w-5 h-5 text-gradient-accent-1 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Computing derivatives for network training
                  </p>
                </Link>

                <Link
                  href="/learn/neural-networks/backpropagation-in-action"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-gradient-accent-3/50 hover:shadow-xl hover:shadow-gradient-accent-3/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-cyan-400 transition-colors">
                      <span className="mr-2">6.</span>Backpropagation in Action
                    </h3>
                    <svg className="w-5 h-5 text-gradient-accent-3 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Understanding the backpropagation algorithm
                  </p>
                </Link>

                <Link
                  href="/learn/neural-networks/implementing-backpropagation"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-teal-500/50 hover:shadow-xl hover:shadow-teal-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-teal-400 transition-colors">
                      <span className="mr-2">7.</span>Implementing Backpropagation
                    </h3>
                    <svg className="w-5 h-5 text-teal-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Coding the backpropagation algorithm from scratch
                  </p>
                </Link>
              </div>
            </div>

            {/* Attention Mechanism Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-red-500 to-pink-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-foreground">
                    Attention Mechanism
                  </h2>
                  <p className="text-muted-foreground">
                    Understanding attention and self-attention
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link
                  href="/learn/attention-mechanism/what-is-attention"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-red-500/50 hover:shadow-xl hover:shadow-red-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-red-400 transition-colors">
                      <span className="mr-2">1.</span>What is Attention
                    </h3>
                    <svg className="w-5 h-5 text-red-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Understanding the attention mechanism
                  </p>
                </Link>

                <Link
                  href="/learn/attention-mechanism/self-attention-from-scratch"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-pink-500/50 hover:shadow-xl hover:shadow-pink-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-pink-400 transition-colors">
                      <span className="mr-2">2.</span>Self Attention from Scratch
                    </h3>
                    <svg className="w-5 h-5 text-pink-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Building self-attention from the ground up
                  </p>
                </Link>

                <Link
                  href="/learn/attention-mechanism/calculating-attention-scores"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-rose-500/50 hover:shadow-xl hover:shadow-rose-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-rose-400 transition-colors">
                      <span className="mr-2">3.</span>Calculating Attention Scores
                    </h3>
                    <svg className="w-5 h-5 text-rose-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Computing query-key-value similarities
                  </p>
                </Link>

                <Link
                  href="/learn/attention-mechanism/applying-attention-weights"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-fuchsia-500/50 hover:shadow-xl hover:shadow-fuchsia-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-fuchsia-400 transition-colors">
                      <span className="mr-2">4.</span>Applying Attention Weights
                    </h3>
                    <svg className="w-5 h-5 text-fuchsia-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Using attention scores to weight values
                  </p>
                </Link>

                <Link
                  href="/learn/attention-mechanism/multi-head-attention"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-gradient-accent-2/50 hover:shadow-xl hover:shadow-gradient-accent-2/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-purple-400 transition-colors">
                      <span className="mr-2">5.</span>Multi Head Attention
                    </h3>
                    <svg className="w-5 h-5 text-gradient-accent-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Parallel attention mechanisms
                  </p>
                </Link>

                <Link
                  href="/learn/attention-mechanism/attention-in-code"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-violet-500/50 hover:shadow-xl hover:shadow-violet-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-violet-400 transition-colors">
                      <span className="mr-2">6.</span>Attention in Code
                    </h3>
                    <svg className="w-5 h-5 text-violet-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Implementing attention mechanisms in Python
                  </p>
                </Link>
              </div>
            </div>

            {/* Transformer Feedforward Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-foreground">
                    Transformer Feedforward
                  </h2>
                  <p className="text-muted-foreground">
                    Feedforward networks and Mixture of Experts
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link
                  href="/learn/transformer-feedforward/the-feedforward-layer"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-gradient-accent-1/50 hover:shadow-xl hover:shadow-gradient-accent-1/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-blue-400 transition-colors">
                      <span className="mr-2">1.</span>The Feedforward Layer
                    </h3>
                    <svg className="w-5 h-5 text-gradient-accent-1 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Understanding transformer feedforward networks
                  </p>
                </Link>

                <Link
                  href="/learn/transformer-feedforward/what-is-mixture-of-experts"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-indigo-500/50 hover:shadow-xl hover:shadow-indigo-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-indigo-400 transition-colors">
                      <span className="mr-2">2.</span>What is Mixture of Experts
                    </h3>
                    <svg className="w-5 h-5 text-indigo-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Introduction to MoE architecture
                  </p>
                </Link>

                <Link
                  href="/learn/transformer-feedforward/the-expert"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-gradient-accent-3/50 hover:shadow-xl hover:shadow-gradient-accent-3/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-cyan-400 transition-colors">
                      <span className="mr-2">3.</span>The Expert
                    </h3>
                    <svg className="w-5 h-5 text-gradient-accent-3 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Understanding individual expert networks
                  </p>
                </Link>

                <Link
                  href="/learn/transformer-feedforward/the-gate"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-teal-500/50 hover:shadow-xl hover:shadow-teal-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-teal-400 transition-colors">
                      <span className="mr-2">4.</span>The Gate
                    </h3>
                    <svg className="w-5 h-5 text-teal-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Routing and gating mechanisms in MoE
                  </p>
                </Link>

                <Link
                  href="/learn/transformer-feedforward/combining-experts"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-sky-500/50 hover:shadow-xl hover:shadow-sky-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-sky-400 transition-colors">
                      <span className="mr-2">5.</span>Combining Experts
                    </h3>
                    <svg className="w-5 h-5 text-sky-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Merging multiple expert outputs
                  </p>
                </Link>

                <Link
                  href="/learn/transformer-feedforward/moe-in-a-transformer"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-emerald-500/50 hover:shadow-xl hover:shadow-emerald-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-emerald-400 transition-colors">
                      <span className="mr-2">6.</span>MoE in a Transformer
                    </h3>
                    <svg className="w-5 h-5 text-emerald-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Integrating mixture of experts in transformers
                  </p>
                </Link>

                <Link
                  href="/learn/transformer-feedforward/moe-in-code"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-green-500/50 hover:shadow-xl hover:shadow-green-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-green-400 transition-colors">
                      <span className="mr-2">7.</span>MoE in Code
                    </h3>
                    <svg className="w-5 h-5 text-green-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Implementing mixture of experts in Python
                  </p>
                </Link>

                <Link
                  href="/learn/transformer-feedforward/the-deepseek-mlp"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-lime-500/50 hover:shadow-xl hover:shadow-lime-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-lime-400 transition-colors">
                      <span className="mr-2">8.</span>The DeepSeek MLP
                    </h3>
                    <svg className="w-5 h-5 text-lime-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    DeepSeek&apos;s advanced MLP architecture
                  </p>
                </Link>
              </div>
            </div>

            {/* Building a Transformer Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-foreground">
                    Building a Transformer
                  </h2>
                  <p className="text-muted-foreground">
                    Complete transformer implementation from scratch
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link
                  href="/learn/building-a-transformer/transformer-architecture"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-orange-500/50 hover:shadow-xl hover:shadow-orange-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-orange-400 transition-colors">
                      <span className="mr-2">1.</span>Transformer Architecture
                    </h3>
                    <svg className="w-5 h-5 text-orange-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Understanding the complete transformer structure
                  </p>
                </Link>

                <Link
                  href="/learn/building-a-transformer/rope-positional-encoding"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-amber-500/50 hover:shadow-xl hover:shadow-amber-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-amber-400 transition-colors">
                      <span className="mr-2">2.</span>RoPE Positional Encoding
                    </h3>
                    <svg className="w-5 h-5 text-amber-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Rotary position embeddings for transformers
                  </p>
                </Link>

                <Link
                  href="/learn/building-a-transformer/building-a-transformer-block"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-yellow-500/50 hover:shadow-xl hover:shadow-yellow-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-yellow-400 transition-colors">
                      <span className="mr-2">3.</span>Building a Transformer Block
                    </h3>
                    <svg className="w-5 h-5 text-yellow-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Constructing individual transformer layers
                  </p>
                </Link>

                <Link
                  href="/learn/building-a-transformer/the-final-linear-layer"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-red-500/50 hover:shadow-xl hover:shadow-red-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-red-400 transition-colors">
                      <span className="mr-2">4.</span>The Final Linear Layer
                    </h3>
                    <svg className="w-5 h-5 text-red-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Output projection and prediction head
                  </p>
                </Link>

                <Link
                  href="/learn/building-a-transformer/full-transformer-in-code"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-pink-500/50 hover:shadow-xl hover:shadow-pink-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-pink-400 transition-colors">
                      <span className="mr-2">5.</span>Full Transformer in Code
                    </h3>
                    <svg className="w-5 h-5 text-pink-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Complete transformer implementation
                  </p>
                </Link>

                <Link
                  href="/learn/building-a-transformer/training-a-transformer"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-rose-500/50 hover:shadow-xl hover:shadow-rose-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-rose-400 transition-colors">
                      <span className="mr-2">6.</span>Training a Transformer
                    </h3>
                    <svg className="w-5 h-5 text-rose-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Training process and optimization
                  </p>
                </Link>
              </div>
            </div>

            {/* Large Language Models Module */}
            <div className="bg-gradient-to-br from-slate-800/50 to-slate-700/50 backdrop-blur-sm border border-slate-600/50 rounded-2xl p-8">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-3xl font-bold text-foreground">
                    Large Language Models
                  </h2>
                  <p className="text-muted-foreground">
                    Understanding LLM training and optimization
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Link
                  href="/learn/large-language-models/batch-size-vs-sequence-length"
                  className="group bg-card/50 border border-border/50 rounded-xl p-6 hover:border-emerald-500/50 hover:shadow-xl hover:shadow-emerald-500/10 transition-all duration-300"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-xl font-semibold text-foreground group-hover:text-gradient-accent-emerald-400 transition-colors">
                      <span className="mr-2">1.</span>Batch Size vs Sequence Length
                    </h3>
                    <svg className="w-5 h-5 text-emerald-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <p className="text-muted-foreground text-sm">
                    Understanding the trade-offs between batch size and sequence length
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

