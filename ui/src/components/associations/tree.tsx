import React, { useState, useRef } from 'react';
import { asyncDataLoaderFeature, hotkeysCoreFeature, selectionFeature } from '@headless-tree/core';
import { useTree } from '@headless-tree/react';
import cn from 'classnames';
import styled from 'styled-components';

import { Spinner } from 'react-bootstrap';
import { ErrorBoundary } from 'react-error-boundary';

// Project imports
import { RenderErrorAlert } from '@components';
import { getInitialTree, growTree } from '@thorpi';
import { Seed, Graph, BranchNode, Direction } from '@models';
import { getNodeName } from './utilities';

import ArrowSVG from '@assets/icons/arrow_drop_up.svg';
export const ArrowIcon = ArrowSVG.replace('REPLACEME', '64cc66');

const Tree = styled.div`
  .tree button[role='treeitem'] {
    display: flex;
    background: transparent;
    border: none;
    width: 100%;
    padding: 0 0 2px 0;
  }

  .treeitem {
    width: 100%;
    text-align: left;
    //background-color: white;
    color: var(--thorium-text);
    padding: 6px 10px;
    position: relative;
    border-radius: 8px;
    transition: background-color 0.2s ease;
    cursor: pointer;
  }
  .treeitem:hover {
    background-color: rgb(0, 102, 255, 0.1);
    color: var(--thorium-text);
  }

  .treeitem:hover {
    border-color: black;
  }

  .tree button[role='treeitem']:focus {
    outline: none;
  }

  .treeitem.selected {
    //background-color: #eee;
  }

  button:focus-visible .treeitem.focused,
  .treeitem.searchmatch.focused {
    outline: 2px solid black;
  }

  .treeitem.drop {
    border-color: var(--selected-color);
    background-color: #e1f1f8;
  }

  .treeitem.searchmatch {
    background-color: #e1f8ff;
  }

  .treeitem.folder:before {
    content: url(data:image/svg+xml;base64,PHN2ZyB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IiB2aWV3Qm94PSIwIDAgMTYgMTYiIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDE2IDE2IiB4bWw6c3BhY2U9InByZXNlcnZlIj48Zz48Zz48cGF0aCBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTQuNjQ2IDEuNjQ2YS41LjUgMCAwIDEgLjcwOCAwbDYgNmEuNS41IDAgMCAxIDAgLjcwOGwtNiA2YS41LjUgMCAwIDEtLjcwOC0uNzA4TDEwLjI5MyA4IDQuNjQ2IDIuMzU0YS41LjUgMCAwIDEgMC0uNzA4eiIgY2xhc3M9InJjdC10cmVlLWl0ZW0tYXJyb3ctcGF0aCI+PC9wYXRoPjwvZz48L2c+PC9zdmc+);
    background-color: white;
    width: 10px;
    display: inline-block;
    z-index: 1;
    margin-right: 4px;
    transition: transform 0.1s ease-in-out;
  }

  .treeitem.folder.expanded:before {
    transform: rotate(90deg);
  }

  .treeitem:not(.folder) {
    padding-left: 24px;
  }

  .treeitem.selected:after {
    content: ' ';
    position: absolute;
    top: 5px;
    left: -2px;
    height: 24px;
    width: 4px;
    background-color: #0366d6;
    border-radius: 99px;
  }

  .description {
    font-family: sans-serif;
    font-size: 0.8rem;
    background-color: #eee;
    border-radius: 8px;
    padding: 8px 12px;
  }

  .outeritem {
    display: flex;
    align-items: center;
    gap: 2px;
  }
  .outeritem button:not([role='treeitem']) {
    padding: 2px 4px;
    height: 80%;
  }
`;

async function getRootData(initial: Seed) {
  return await getInitialTree(initial, true, 1, console.log).then((data) => {
    if (data) {
      return data;
    }
    return null;
  });
}

// Grow the graph from a single node
async function growGraph(nodeId: string, graph: Graph) {
  // check if graph is valid and id is set
  return await growTree(graph.id, [nodeId], console.log).then((data) => {
    const newGraph = structuredClone(graph);
    // update node map
    if (data?.data_map) {
      Object.keys(data.data_map).map((node) => {
        if (!(node in newGraph.data_map)) {
          newGraph.data_map[node] = data.data_map[node];
        }
      });
    }
    // update growable
    const newGrowable = newGraph.growable.filter((item) => nodeId != item);
    if (data?.growable) {
      data.growable.forEach((item) => newGrowable.push(item));
    }
    newGraph.growable = newGrowable;
    // update branches
    if (data?.branches) {
      Object.keys(data.branches).map((source) => {
        // branch source is already present, so maybe we are expanding that source
        if (source in newGraph.branches) {
          newGraph.branches[source].push(...data.branches[source]);
          // branch root not present, this is a new branch source
        } else {
          newGraph.branches[source] = data.branches[source];
        }
      });
    }
    return newGraph;
  });
}

interface AssociationTreeProps {
  initial: Seed; // Thorium tree data object
}

// Browsable Tree component
const AssociationTree: React.FC<AssociationTreeProps> = ({ initial }) => {
  const graph = useRef<Graph | null>(null);
  const [loadingItemData, setLoadingItemData] = useState<string[]>([]);
  const [loadingItemChildrens, setLoadingItemChildrens] = useState<string[]>([]);
  const tree = useTree<string>({
    state: { loadingItemData, loadingItemChildrens },
    setLoadingItemData,
    setLoadingItemChildrens,
    rootItemId: 'root',
    getItemName: (node) => {
      const nodeId = node.getId();
      if (graph.current?.data_map && nodeId in graph.current.data_map) {
        // TODO: need to figure out if the node is the parent or child of current sample
        let nodeName = getNodeName(graph.current.data_map[nodeId], 100);

        //if (checkIsParent(nodeId, graph.current)) return nodeName + ' (parent)';
        return nodeName;
      }
      return node.getItemData();
    },
    isItemFolder: (node) => {
      const nodeId = node.getId();
      const branches = graph.current?.branches ? Object.keys(graph.current.branches) : [];
      if (graph.current?.growable?.includes(nodeId) || branches.includes(nodeId)) {
        return true;
      }
      return false;
    },
    createLoadingItemData: () => 'loading...',
    dataLoader: {
      getItem: (nodeId) => nodeId,
      getChildren: async (nodeId) => {
        const children: string[] = [];
        // root is a hard coded value to initialize the tree, its not a real tree node
        if (nodeId == 'root') {
          const initialGraph = await getRootData(initial);
          if (initialGraph) {
            graph.current = initialGraph;
            return initialGraph.initial;
          }
          // expand an existing node
        } else if (graph.current?.growable.includes(nodeId)) {
          // we need to expand this node with a fetch
          const updatedGraph = await growGraph(nodeId, graph.current);
          // now return any children for this node
          if (Object.keys(updatedGraph.branches).includes(nodeId)) {
            updatedGraph.branches[nodeId].forEach((child) => children.push(child.node));
          }
          graph.current = updatedGraph;
          // node already exists, probably because we prepopulated it ahead of time
        } else if (graph.current?.branches && Object.keys(graph.current.branches).includes(nodeId)) {
          graph.current.branches[nodeId].forEach((node: BranchNode) => children.push(node.node));
        }
        return children;
      },
    },
    indent: 20,
    features: [asyncDataLoaderFeature, selectionFeature, hotkeysCoreFeature],
  });

  return (
    <>
      <Tree>
        <div {...tree.getContainerProps()} className="tree">
          {tree.getItems().map((item) => (
            <button {...item.getProps()} key={item.getId()} style={{ paddingLeft: `${item.getItemMeta().level * 20}px` }}>
              <div
                className={cn('treeitem pt-1 pb-0', {
                  focused: item.isFocused(),
                  expanded: item.isExpanded(),
                  selected: item.isSelected(),
                  folder: item.isFolder(),
                })}
              >
                {item.getItemName()}
                {item.isLoading() && <Spinner className="m-4 loading" animation="border" />}
              </div>
            </button>
          ))}
        </div>
      </Tree>
    </>
  );
};

interface AssociationTreeWrapperProps {
  initial: Seed; // some initial root graph nodes
}

export const AssociationTreeWrapper: React.FC<AssociationTreeWrapperProps> = ({ initial }) => {
  return (
    <ErrorBoundary fallback={<RenderErrorAlert page={false} />}>
      <AssociationTree initial={initial} />
    </ErrorBoundary>
  );
};

export default AssociationTreeWrapper;
