import { useEffect, useState } from 'react';
import type { FC } from 'react';
import { Button, Modal } from 'react-bootstrap';

// project imports
import { pipelineChecker } from './checker';
import AlertBanner from '@components/shared/alerts/AlertBanner';
import ImagePipelineEditor from '@components/shared/inputs/code/CodeEditor/ImagePipelineEditor';
import FormatToggle from '@components/shared/inputs/code/CodeEditor/FormatToggle';
import { listImages } from '@thorpi/images';
import { createPipeline } from '@thorpi/pipelines';
import { FormatType } from '@utilities/rules/types';
import { editorObjectToPipelineCreate } from '@utilities/transforms/pipeline';
import type { PipelineCreate } from '@models/pipelines';

export const PIPELINE_CREATE_TEMPLATE: Record<string, unknown> = {
  group: '',
  name: '',
  order: [],
  sla: 604800,
  description: '',
};

interface CreatePipelineModalProps {
  show: boolean;
  onHide: () => void;
  onCreated: () => void;
  initialData: Record<string, unknown>;
}

const CreatePipelineModal: FC<CreatePipelineModalProps> = ({ show, onHide, onCreated, initialData }) => {
  const [pipelineObj, setPipelineObj] = useState<Record<string, unknown>>(initialData);
  const [format, setFormat] = useState<FormatType>(FormatType.YAML);
  const [parseValid, setParseValid] = useState(false);
  const [createError, setCreateError] = useState('');

  useEffect(() => {
    if (show) {
      setPipelineObj(initialData);
      setParseValid(false);
      setCreateError('');
    } else {
      pipelineChecker.clearValidImageNames();
    }
  }, [show, initialData]);

  const editorGroup = typeof pipelineObj.group === 'string' ? pipelineObj.group : '';

  useEffect(() => {
    if (!editorGroup) return;
    let cancelled = false;
    void listImages(editorGroup, () => {}, false, null, 1000).then((result) => {
      if (cancelled) return;
      if (result && 'names' in result) pipelineChecker.setValidImageNames(editorGroup, result.names);
    });
    return () => {
      cancelled = true;
    };
  }, [editorGroup]);

  const handleEditorChange = (obj: Record<string, unknown> | null) => {
    if (obj) {
      setPipelineObj(obj);
      setParseValid(true);
    } else {
      setParseValid(false);
    }
  };

  async function handlePipelineCreate() {
    const data = editorObjectToPipelineCreate(pipelineObj);
    if (!data) {
      setCreateError('Pipeline group, name, and order are required');
      return;
    }
    if (await createPipeline(data as PipelineCreate, setCreateError)) {
      onHide();
      onCreated();
    }
  }

  return (
    <Modal show={show} onHide={onHide} backdrop="static" keyboard={false} size="lg" centered>
      <Modal.Header closeButton>
        <Modal.Title>Create New Pipeline</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <div className="mb-3">
          <FormatToggle format={format} onFormatChange={setFormat} />
        </div>
        <ImagePipelineEditor value={pipelineObj} onChange={handleEditorChange} checker={pipelineChecker} format={format} height="350px" />
        {createError !== '' && <AlertBanner className="mt-4">{createError}</AlertBanner>}
      </Modal.Body>
      <Modal.Footer className="d-flex justify-content-center">
        <Button className="ok-btn" disabled={!parseValid} onClick={() => void handlePipelineCreate()}>
          Create
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default CreatePipelineModal;
