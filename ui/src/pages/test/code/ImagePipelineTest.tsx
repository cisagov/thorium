import React, { useState, useEffect, useMemo } from 'react';
import ImagePipelineEditor from '@components/shared/inputs/code/CodeEditor/ImagePipelineEditor';
import FormatToggle from '@components/shared/inputs/code/CodeEditor/FormatToggle';
import Page from '@components/pages/Page';
import { addPreview, clearPreview, acceptPreview } from '@components/shared/inputs/code/CodeEditor/SuggestionPreview';
import { ImageChecker } from '@utilities/rules/tools/image';
import { PipelineChecker } from '@utilities/rules/tools/pipeline';
import { FormatType } from '@utilities/rules/types';
import styled from 'styled-components';

const SAMPLE_IMAGE: Record<string, unknown> = {
  group: 'analysis',
  name: 'yara-scanner',
  image: 'thorium/yara-scanner:latest',
  scaler: 'K8s',
  timeout: 300,
  lifetime: {
    counter: 'jobs',
    amount: 32,
  },
  resources: {
    cpu: 1000,
    memory: 512,
    ephemeral_storage: 1024,
  },
  display_type: 'JSON',
  collect_logs: true,
  generator: false,
  args: {
    output: 'Append',
    output_files: 'None',
  },
  dependencies: {
    samples: {
      location: '/tmp/thorium/samples',
      strategy: 'Paths',
      naming: 'Sha256',
    },
  },
  output_collection: {
    handler: 'Files',
    files: {
      results: '/tmp/thorium/results',
      result_files: '/tmp/thorium/result-files',
    },
  },
  description: 'Scans files with YARA rules for pattern matching',
};

const SAMPLE_PIPELINE: Record<string, unknown> = {
  group: 'analysis',
  name: 'triage',
  order: ['file-info', ['yara-scanner', 'clamav'], 'report-generator'],
  sla: 604800,
  description: 'Standard triage pipeline for submitted files',
};

const TabBar = styled.div`
  display: flex;
  gap: 0;
  margin-bottom: 0;
`;

const Tab = styled.button<{ $active: boolean }>`
  padding: 8px 20px;
  border: 1px solid var(--thorium-panel-border);
  border-bottom: ${(props) => (props.$active ? 'none' : '1px solid var(--thorium-panel-border)')};
  background: ${(props) => (props.$active ? 'var(--thorium-panel-bg)' : 'var(--thorium-highlight-panel-bg)')};
  color: var(--thorium-text);
  cursor: pointer;
  font-size: 13px;
  font-weight: ${(props) => (props.$active ? 600 : 400)};
  border-radius: 4px 4px 0 0;
  margin-bottom: -1px;
  position: relative;
  z-index: ${(props) => (props.$active ? 1 : 0)};

  &:hover {
    background: ${(props) => (props.$active ? 'var(--thorium-panel-bg)' : 'var(--thorium-info-secondary-bg)')};
  }
`;

type EditorTab = 'image' | 'pipeline';

const imageChecker = new ImageChecker();
const pipelineChecker = new PipelineChecker();

const ImagePipelineTest: React.FC = () => {
  const [activeTab, setActiveTab] = useState<EditorTab>('image');
  const [format, setFormat] = useState<FormatType>(FormatType.YAML);
  const [imageValue, setImageValue] = useState<Record<string, unknown>>(SAMPLE_IMAGE);
  const [pipelineValue, setPipelineValue] = useState<Record<string, unknown>>(SAMPLE_PIPELINE);
  const [parsedOutput, setParsedOutput] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    (window as unknown as Record<string, unknown>)['__imagePipelineTestHelpers'] = {
      addPreview,
      clearPreview,
      acceptPreview,
    };
    return () => {
      delete (window as unknown as Record<string, unknown>)['__imagePipelineTestHelpers'];
    };
  }, []);

  const currentValue = activeTab === 'image' ? imageValue : pipelineValue;
  const currentChecker = activeTab === 'image' ? imageChecker : pipelineChecker;

  const handleChange = useMemo(() => {
    return (obj: Record<string, unknown> | null) => {
      setParsedOutput(obj);
      if (obj) {
        if (activeTab === 'image') {
          setImageValue(obj);
        } else {
          setPipelineValue(obj);
        }
      }
    };
  }, [activeTab]);

  return (
    <Page title="Image/Pipeline Editor Test">
      <h2>Image/Pipeline Editor Test</h2>

      <div style={{ marginBottom: 12 }}>
        <FormatToggle format={format} onFormatChange={setFormat} />
      </div>

      <TabBar>
        <Tab $active={activeTab === 'image'} onClick={() => setActiveTab('image')}>
          Image
        </Tab>
        <Tab $active={activeTab === 'pipeline'} onClick={() => setActiveTab('pipeline')}>
          Pipeline
        </Tab>
      </TabBar>

      <ImagePipelineEditor
        key={`${activeTab}-${format}`}
        value={currentValue}
        onChange={handleChange}
        checker={currentChecker}
        format={format}
        height="500px"
      />

      <pre data-testid="parsed-output" style={{ marginTop: 16, fontSize: 11, opacity: 0.7 }}>
        {parsedOutput ? JSON.stringify(parsedOutput, null, 2) : '(no valid parse)'}
      </pre>
    </Page>
  );
};

export default ImagePipelineTest;
