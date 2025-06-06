// download SVG of graph
export const exportSVG = (id: string, cy: React.RefObject<cytoscape.Core | null>) => {
  if (cy?.current) {
    //@ts-ignore // checker doesn't recognize loaded plugins
    const svg = cy.current.svg({ bg: getComputedStyle(document.documentElement).getPropertyValue('--thorium-panel-bg') });
    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${id}-graph.svg`;
    a.click();
    URL.revokeObjectURL(url); // Clean up the URL object
  }
};

// download PNG image of graph
export const exportPNG = (id: string, cy: React.RefObject<cytoscape.Core | null>) => {
  if (cy?.current) {
    const png = cy.current.png({ bg: getComputedStyle(document.documentElement).getPropertyValue('--thorium-panel-bg') });
    const a = document.createElement('a');
    a.href = png;
    a.download = `${id}-graph.png`;
    a.click();
  }
};

// download JPEG image of graph
export const exportJPEG = (id: string, cy: React.RefObject<cytoscape.Core | null>) => {
  if (cy?.current) {
    const jpeg = cy.current.jpeg({ bg: getComputedStyle(document.documentElement).getPropertyValue('--thorium-panel-bg') });
    const a = document.createElement('a');
    a.href = jpeg;
    a.download = `${id}-graph.jpeg`;
    a.click();
  }
};
