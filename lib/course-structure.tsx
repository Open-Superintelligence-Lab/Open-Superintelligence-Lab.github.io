export interface LessonItem {
  title: string;
  titleZh: string;
  href: string;
}

export interface ModuleData {
  title: string;
  titleZh: string;
  icon: React.ReactNode;
  lessons: LessonItem[];
}

export const getCourseModules = (): ModuleData[] => [
  {
    title: "Mathematics Fundamentals",
    titleZh: "数学基础",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
      </svg>
    ),
    lessons: [
      {
        title: "Functions",
        titleZh: "函数",
        href: "/learn/math/functions"
      },
      {
        title: "Derivatives",
        titleZh: "导数",
        href: "/learn/math/derivatives"
      },
      {
        title: "Vectors",
        titleZh: "向量",
        href: "/learn/math/vectors"
      },
      {
        title: "Matrices",
        titleZh: "矩阵",
        href: "/learn/math/matrices"
      },
      {
        title: "Gradients",
        titleZh: "梯度",
        href: "/learn/math/gradients"
      }
    ]
  },
  {
    title: "PyTorch Fundamentals",
    titleZh: "PyTorch基础",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
      </svg>
    ),
    lessons: [
      {
        title: "Creating Tensors",
        titleZh: "创建张量",
        href: "/learn/tensors/creating-tensors"
      },
      {
        title: "Tensor Addition",
        titleZh: "张量加法",
        href: "/learn/tensors/tensor-addition"
      },
      {
        title: "Matrix Multiplication",
        titleZh: "矩阵乘法",
        href: "/learn/tensors/matrix-multiplication"
      },
      {
        title: "Transposing Tensors",
        titleZh: "张量转置",
        href: "/learn/tensors/transposing-tensors"
      },
      {
        title: "Reshaping Tensors",
        titleZh: "张量重塑",
        href: "/learn/tensors/reshaping-tensors"
      },
      {
        title: "Indexing and Slicing",
        titleZh: "索引和切片",
        href: "/learn/tensors/indexing-and-slicing"
      },
      {
        title: "Concatenating Tensors",
        titleZh: "张量拼接",
        href: "/learn/tensors/concatenating-tensors"
      },
      {
        title: "Creating Special Tensors",
        titleZh: "创建特殊张量",
        href: "/learn/tensors/creating-special-tensors"
      }
    ]
  },
  {
    title: "Neuron From Scratch",
    titleZh: "从零开始构建神经元",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
    lessons: [
      {
        title: "What is a Neuron",
        titleZh: "什么是神经元",
        href: "/learn/neuron-from-scratch/what-is-a-neuron"
      },
      {
        title: "The Linear Step",
        titleZh: "线性步骤",
        href: "/learn/neuron-from-scratch/the-linear-step"
      },
      {
        title: "The Activation Function",
        titleZh: "激活函数",
        href: "/learn/neuron-from-scratch/the-activation-function"
      },
      {
        title: "Building a Neuron in Python",
        titleZh: "用Python构建神经元",
        href: "/learn/neuron-from-scratch/building-a-neuron-in-python"
      },
      {
        title: "Making a Prediction",
        titleZh: "进行预测",
        href: "/learn/neuron-from-scratch/making-a-prediction"
      },
      {
        title: "The Concept of Loss",
        titleZh: "损失概念",
        href: "/learn/neuron-from-scratch/the-concept-of-loss"
      },
      {
        title: "The Concept of Learning",
        titleZh: "学习概念",
        href: "/learn/neuron-from-scratch/the-concept-of-learning"
      }
    ]
  },
  {
    title: "Activation Functions",
    titleZh: "激活函数",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    lessons: [
      {
        title: "ReLU",
        titleZh: "ReLU",
        href: "/learn/activation-functions/relu"
      },
      {
        title: "Sigmoid",
        titleZh: "Sigmoid",
        href: "/learn/activation-functions/sigmoid"
      },
      {
        title: "Tanh",
        titleZh: "Tanh",
        href: "/learn/activation-functions/tanh"
      },
      {
        title: "SiLU",
        titleZh: "SiLU",
        href: "/learn/activation-functions/silu"
      },
      {
        title: "SwiGLU",
        titleZh: "SwiGLU",
        href: "/learn/activation-functions/swiglu"
      },
      {
        title: "Softmax",
        titleZh: "Softmax",
        href: "/learn/activation-functions/softmax"
      }
    ]
  },
  {
    title: "Neural Networks from Scratch",
    titleZh: "从零开始的神经网络",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    lessons: [
      {
        title: "Architecture of a Network",
        titleZh: "网络架构",
        href: "/learn/neural-networks/architecture-of-a-network"
      },
      {
        title: "Building a Layer",
        titleZh: "构建层",
        href: "/learn/neural-networks/building-a-layer"
      },
      {
        title: "Implementing a Network",
        titleZh: "实现网络",
        href: "/learn/neural-networks/implementing-a-network"
      },
      {
        title: "The Chain Rule",
        titleZh: "链式法则",
        href: "/learn/neural-networks/the-chain-rule"
      },
      {
        title: "Calculating Gradients",
        titleZh: "计算梯度",
        href: "/learn/neural-networks/calculating-gradients"
      },
      {
        title: "Backpropagation in Action",
        titleZh: "反向传播实战",
        href: "/learn/neural-networks/backpropagation-in-action"
      },
      {
        title: "Implementing Backpropagation",
        titleZh: "实现反向传播",
        href: "/learn/neural-networks/implementing-backpropagation"
      }
    ]
  },
  {
    title: "Attention Mechanism",
    titleZh: "注意力机制",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
      </svg>
    ),
    lessons: [
      {
        title: "What is Attention",
        titleZh: "什么是注意力",
        href: "/learn/attention-mechanism/what-is-attention"
      },
      {
        title: "Self Attention from Scratch",
        titleZh: "从零开始自注意力",
        href: "/learn/attention-mechanism/self-attention-from-scratch"
      },
      {
        title: "Calculating Attention Scores",
        titleZh: "计算注意力分数",
        href: "/learn/attention-mechanism/calculating-attention-scores"
      },
      {
        title: "Applying Attention Weights",
        titleZh: "应用注意力权重",
        href: "/learn/attention-mechanism/applying-attention-weights"
      },
      {
        title: "Multi Head Attention",
        titleZh: "多头注意力",
        href: "/learn/attention-mechanism/multi-head-attention"
      },
      {
        title: "Attention in Code",
        titleZh: "注意力代码实现",
        href: "/learn/attention-mechanism/attention-in-code"
      }
    ]
  },
  {
    title: "Transformer Feedforward",
    titleZh: "Transformer前馈网络",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    ),
    lessons: [
      {
        title: "The Feedforward Layer",
        titleZh: "前馈层",
        href: "/learn/transformer-feedforward/the-feedforward-layer"
      },
      {
        title: "What is Mixture of Experts",
        titleZh: "什么是专家混合",
        href: "/learn/transformer-feedforward/what-is-mixture-of-experts"
      },
      {
        title: "The Expert",
        titleZh: "专家",
        href: "/learn/transformer-feedforward/the-expert"
      },
      {
        title: "The Gate",
        titleZh: "门控",
        href: "/learn/transformer-feedforward/the-gate"
      },
      {
        title: "Combining Experts",
        titleZh: "组合专家",
        href: "/learn/transformer-feedforward/combining-experts"
      },
      {
        title: "MoE in a Transformer",
        titleZh: "Transformer中的MoE",
        href: "/learn/transformer-feedforward/moe-in-a-transformer"
      },
      {
        title: "MoE in Code",
        titleZh: "MoE代码实现",
        href: "/learn/transformer-feedforward/moe-in-code"
      },
      {
        title: "The DeepSeek MLP",
        titleZh: "DeepSeek MLP",
        href: "/learn/transformer-feedforward/the-deepseek-mlp"
      }
    ]
  },
  {
    title: "Building a Transformer",
    titleZh: "构建Transformer",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
      </svg>
    ),
    lessons: [
      {
        title: "Transformer Architecture",
        titleZh: "Transformer架构",
        href: "/learn/building-a-transformer/transformer-architecture"
      },
      {
        title: "RoPE Positional Encoding",
        titleZh: "RoPE位置编码",
        href: "/learn/building-a-transformer/rope-positional-encoding"
      },
      {
        title: "Building a Transformer Block",
        titleZh: "构建Transformer块",
        href: "/learn/building-a-transformer/building-a-transformer-block"
      },
      {
        title: "The Final Linear Layer",
        titleZh: "最终线性层",
        href: "/learn/building-a-transformer/the-final-linear-layer"
      },
      {
        title: "Full Transformer in Code",
        titleZh: "完整Transformer代码",
        href: "/learn/building-a-transformer/full-transformer-in-code"
      },
      {
        title: "Training a Transformer",
        titleZh: "训练Transformer",
        href: "/learn/building-a-transformer/training-a-transformer"
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

