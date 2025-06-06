import React from 'react';
import { Button, Form, Dropdown, ButtonGroup } from 'react-bootstrap';
import cytoscape from 'cytoscape';
import styled from 'styled-components';

// project imports
import { spacers } from '@styles';
import { getLayout, GraphLayout } from './layout';
import { exportJPEG, exportPNG } from './export';
import { Edge } from './edges';
import { GraphElement } from './graph';
import { Node } from './nodes';

export interface GraphControls {
  filterChildless: boolean; // don't show childless nodes
  depth: number; // depth of initial graph
  showEdgeLabels: boolean; // display text edge labels
  showNodeLabels: boolean; // display text node labels, differing depending on node type
  selectedElement: Node | Edge | null; // selected graph element whether node or edge
  showNodeInfo: boolean; // show node overview when a node selected
  autoRunLayout: boolean; // run layout after growing graph, only recommended for small graphs
  layoutAlgorithm: GraphLayout; //select predefined layout algorithm, eg: fcose
}

// controls and their setter state types
export type DisplayAction =
  | { type: 'depth'; state: number }
  | { type: 'filterChildless' | 'showEdgeLabels' | 'showNodeLabels' | 'showNodeInfo'; state: boolean }
  | { type: 'selected'; state: GraphElement | null }
  | { type: 'layoutAlgorithm'; state: GraphLayout };

// graph controls props
type GraphControlsProps = {
  graphId: string;
  controls: GraphControls;
  updateControls: React.ActionDispatch<[action: DisplayAction]>;
  cyInstance: React.RefObject<cytoscape.Core | null>;
};

const Controls = styled.div`
  top: 0px;
  left: 0px;
  z-index: 400;
  display: grid;
  width: 200px;
  grid-auto-flow: column dense;
  align-items: center;
  justify-content: start;
  gap: ${spacers.one};
  margin-left: ${spacers.four};
`;

// stacks the two switches
const ControlGroup = styled.div`
  display: grid;
  row-gap: ${spacers.one};
  justify-items: center;
  // make labels small
  & > label {
    font-size: 0.9rem;
    margin: 0;
  }
`;

export const GraphControls: React.FC<GraphControlsProps> = ({ graphId, controls, updateControls, cyInstance }) => (
  <Controls>
    <ControlGroup>
      <Form.Select
        id="form-select-depth"
        className="secondary-btn"
        value={controls.depth}
        onChange={(e) => updateControls({ type: 'depth', state: parseInt(e.target.value, 10) })}
        style={{ width: '75px' }}
      >
        {Array.from({ length: 10 }, (_, idx) => idx + 1).map((depth) => (
          <option key={depth} value={depth}>
            {depth}
          </option>
        ))}
      </Form.Select>
      <Form.Label htmlFor="form-select-depth" className="mb-3">
        Depth
      </Form.Label>
    </ControlGroup>
    <Button
      variant=""
      className="secondary-btn mb-5"
      onClick={() => {
        cyInstance?.current?.layout(getLayout(controls.layoutAlgorithm)).run();
      }}
    >
      Randomize
    </Button>
    <Dropdown as={ButtonGroup} className="mb-5">
      <Dropdown.Toggle className="secondary-btn" variant="" id="download-dropdown">
        Download
      </Dropdown.Toggle>
      <Dropdown.Menu>
        <Dropdown.Item key="download_png_option" onClick={() => exportPNG(graphId, cyInstance)}>
          PNG
        </Dropdown.Item>
        <Dropdown.Item key="download_jpeg_option" onClick={() => exportJPEG(graphId, cyInstance)}>
          JPEG
        </Dropdown.Item>
      </Dropdown.Menu>
    </Dropdown>
    <ControlGroup>
      <Dropdown as={ButtonGroup} className="mb-5">
        <Dropdown.Toggle className="secondary-btn" variant="" id="download-dropdown">
          Layout ({controls.layoutAlgorithm})
        </Dropdown.Toggle>
        <Dropdown.Menu>
          {Object.values(GraphLayout).map((layout) => (
            <Dropdown.Item onClick={() => updateControls({ type: 'layoutAlgorithm', state: layout })}>{layout}</Dropdown.Item>
          ))}
        </Dropdown.Menu>
      </Dropdown>
    </ControlGroup>
    <ControlGroup className="mb-4">
      <Form.Check
        type="switch"
        id="form-show-node"
        label="Node Labels"
        checked={controls.showNodeLabels}
        onChange={() =>
          updateControls({
            type: 'showNodeLabels',
            state: !controls.showNodeLabels,
          })
        }
      />
      <Form.Check
        type="switch"
        id="form-show-edge"
        label="Edge Labels"
        checked={controls.showEdgeLabels}
        onChange={() =>
          updateControls({
            type: 'showEdgeLabels',
            state: !controls.showEdgeLabels,
          })
        }
      />
    </ControlGroup>
  </Controls>
);
