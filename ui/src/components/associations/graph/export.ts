import type { GraphInstance } from './types';

// spec: ./AssociationGraph.spec.md

const captureDataUrl = (graphInstance: GraphInstance, mimeType: string): string | null => {
  const renderer = graphInstance.renderer();
  const scene = graphInstance.scene();
  const camera = graphInstance.camera();
  if (!renderer || !scene || !camera) return null;
  // Re-render immediately before capture to ensure the buffer contains current frame data
  renderer.render(scene, camera);
  return renderer.domElement.toDataURL(mimeType);
};

const downloadDataUrl = (dataUrl: string, filename: string) => {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  a.click();
};

export const exportPNG = (id: string, graphInstance: GraphInstance | null) => {
  if (!graphInstance) return;
  const dataUrl = captureDataUrl(graphInstance, 'image/png');
  if (dataUrl) downloadDataUrl(dataUrl, `${id}-graph.png`);
};

export const exportJPEG = (id: string, graphInstance: GraphInstance | null) => {
  if (!graphInstance) return;
  const dataUrl = captureDataUrl(graphInstance, 'image/jpeg');
  if (dataUrl) downloadDataUrl(dataUrl, `${id}-graph.jpeg`);
};
