'use client';

import React from 'react';
import Chatbot from '@/components/chatbot';
import Canvas from '@/components/canvas';
import { AppLayout } from '@/components/layout/app-layout';

export default function TestChatbot() {
  return (
    <AppLayout>
      <div className="container mx-auto px-6 py-8">
        <h1 className="text-2xl font-bold mb-6">Test Chatbot Interface</h1>
        <div className="space-y-6">
          <div>
            <h2 className="text-lg font-semibold mb-4">AI Assistant</h2>
            <Chatbot projectId="test-project" projectName="Test Project" />
          </div>
          <div>
            <h2 className="text-lg font-semibold mb-4">Results Canvas</h2>
            <Canvas projectId="test-project" />
          </div>
        </div>
      </div>
    </AppLayout>
  );
}
