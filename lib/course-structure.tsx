export interface LessonItem {
  title: string;
  href: string;
}

export interface ModuleData {
  title: string;
  icon: React.ReactNode;
  lessons: LessonItem[];
}

export const getCourseModules = (): ModuleData[] => [
  {
    title: "Mathematics Fundamentals",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
      </svg>
    ),
    lessons: [
      {
        title: "Functions",
        href: "/learn/math/functions"
      },
      {
        title: "Derivatives",
        href: "/learn/math/derivatives"
      },
      {
        title: "Vectors",
        href: "/learn/math/vectors"
      },
      {
        title: "Matrices",
        href: "/learn/math/matrices"
      },
      {
        title: "Gradients",
        href: "/learn/math/gradients"
      }
    ]
  },
  {
    title: "PyTorch Fundamentals",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
      </svg>
    ),
    lessons: [
      {
        title: "Creating Tensors",
        href: "/learn/tensors/creating-tensors"
      },
      {
        title: "Tensor Addition",
        href: "/learn/tensors/tensor-addition"
      },
      {
        title: "Matrix Multiplication",
        href: "/learn/tensors/matrix-multiplication"
      },
      {
        title: "Transposing Tensors",
        href: "/learn/tensors/transposing-tensors"
      },
      {
        title: "Reshaping Tensors",
        href: "/learn/tensors/reshaping-tensors"
      },
      {
        title: "Indexing and Slicing",
        href: "/learn/tensors/indexing-and-slicing"
      },
      {
        title: "Concatenating Tensors",
        href: "/learn/tensors/concatenating-tensors"
      },
      {
        title: "Creating Special Tensors",
        href: "/learn/tensors/creating-special-tensors"
      }
    ]
  },
  {
    title: "Neuron From Scratch",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    lessons: [
      {
        title: "What is a Neuron",
        href: "/learn/neuron-from-scratch/what-is-a-neuron"
      },
      {
        title: "The Linear Step",
        href: "/learn/neuron-from-scratch/the-linear-step"
      },
      {
        title: "The Activation Function",
        href: "/learn/neuron-from-scratch/the-activation-function"
      },
      {
        title: "Building a Neuron in Python",
        href: "/learn/neuron-from-scratch/building-a-neuron-in-python"
      },
      {
        title: "Making a Prediction",
        href: "/learn/neuron-from-scratch/making-a-prediction"
      },
      {
        title: "The Concept of Loss",
        href: "/learn/neuron-from-scratch/the-concept-of-loss"
      },
      {
        title: "The Concept of Learning",
        href: "/learn/neuron-from-scratch/the-concept-of-learning"
      }
    ]
  },
  {
    title: "Activation Functions",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    lessons: [
      {
        title: "ReLU",
        href: "/learn/activation-functions/relu"
      },
      {
        title: "Sigmoid",
        href: "/learn/activation-functions/sigmoid"
      },
      {
        title: "Tanh",
        href: "/learn/activation-functions/tanh"
      },
      {
        title: "SiLU",
        href: "/learn/activation-functions/silu"
      },
      {
        title: "SwiGLU",
        href: "/learn/activation-functions/swiglu"
      },
      {
        title: "Softmax",
        href: "/learn/activation-functions/softmax"
      }
    ]
  },
  {
    title: "Neural Networks from Scratch",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    lessons: [
      {
        title: "Architecture of a Network",
        href: "/learn/neural-networks/architecture-of-a-network"
      },
      {
        title: "Building a Layer",
        href: "/learn/neural-networks/building-a-layer"
      },
      {
        title: "Implementing a Network",
        href: "/learn/neural-networks/implementing-a-network"
      },
      {
        title: "The Chain Rule",
        href: "/learn/neural-networks/the-chain-rule"
      },
      {
        title: "Calculating Gradients",
        href: "/learn/neural-networks/calculating-gradients"
      },
      {
        title: "Backpropagation in Action",
        href: "/learn/neural-networks/backpropagation-in-action"
      },
      {
        title: "Implementing Backpropagation",
        href: "/learn/neural-networks/implementing-backpropagation"
      }
    ]
  },
  {
    title: "Attention Mechanism",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
      </svg>
    ),
    lessons: [
      {
        title: "What is Attention",
        href: "/learn/attention-mechanism/what-is-attention"
      },
      {
        title: "Self Attention from Scratch",
        href: "/learn/attention-mechanism/self-attention-from-scratch"
      },
      {
        title: "Calculating Attention Scores",
        href: "/learn/attention-mechanism/calculating-attention-scores"
      },
      {
        title: "Applying Attention Weights",
        href: "/learn/attention-mechanism/applying-attention-weights"
      },
      {
        title: "Multi Head Attention",
        href: "/learn/attention-mechanism/multi-head-attention"
      },
      {
        title: "Attention in Code",
        href: "/learn/attention-mechanism/attention-in-code"
      }
    ]
  },
  {
    title: "Transformer Feedforward",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
    lessons: [
      {
        title: "The Feedforward Layer",
        href: "/learn/transformer-feedforward/the-feedforward-layer"
      },
      {
        title: "What is Mixture of Experts",
        href: "/learn/transformer-feedforward/what-is-mixture-of-experts"
      },
      {
        title: "The Expert",
        href: "/learn/transformer-feedforward/the-expert"
      },
      {
        title: "The Gate",
        href: "/learn/transformer-feedforward/the-gate"
      },
      {
        title: "Combining Experts",
        href: "/learn/transformer-feedforward/combining-experts"
      },
      {
        title: "MoE in a Transformer",
        href: "/learn/transformer-feedforward/moe-in-a-transformer"
      },
      {
        title: "MoE in Code",
        href: "/learn/transformer-feedforward/moe-in-code"
      }
    ]
  },
  {
    title: "Building a Transformer",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
      </svg>
    ),
    lessons: [
      {
        title: "Transformer Architecture",
        href: "/learn/building-a-transformer/transformer-architecture"
      },
      {
        title: "RoPE Positional Encoding",
        href: "/learn/building-a-transformer/rope-positional-encoding"
      },
      {
        title: "Building a Transformer Block",
        href: "/learn/building-a-transformer/building-a-transformer-block"
      },
      {
        title: "The Final Linear Layer",
        href: "/learn/building-a-transformer/the-final-linear-layer"
      },
      {
        title: "Full Transformer in Code",
        href: "/learn/building-a-transformer/full-transformer-in-code"
      },
      {
        title: "Training a Transformer",
        href: "/learn/building-a-transformer/training-a-transformer"
      }
    ]
  },
  {
    title: "Large Language Models",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
      </svg>
    ),
    lessons: [
      {
        title: "Batch Size vs Sequence Length",
        href: "/learn/large-language-models/batch-size-vs-sequence-length"
      }
    ]
  }
];

// Get all lessons as a flat array
export const getAllLessons = (): LessonItem[] => {
  const modules = getCourseModules();
  return modules.flatMap(module => module.lessons);
};

// Get next and previous lessons for a given href
export const getAdjacentLessons = (currentHref: string) => {
  const allLessons = getAllLessons();
  const currentIndex = allLessons.findIndex(lesson => lesson.href === currentHref);

  if (currentIndex === -1) {
    return { prev: null, next: null };
  }

  const prev = currentIndex > 0 ? allLessons[currentIndex - 1] : null;
  const next = currentIndex < allLessons.length - 1 ? allLessons[currentIndex + 1] : null;

  return { prev, next };
};

