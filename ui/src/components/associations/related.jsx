import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Alert, Card, Col, Row, Form } from 'react-bootstrap';
import { GraphCanvas } from 'reagraph';
import convert from 'color-convert';

// project imports
import { getFileDetails, getResults } from '@thorpi';
import { Subtitle, TagBadge, filterIncludedTags } from '@components';

const theme = (canvasBg, text, highlightText, secondaryText) => {
  const theme = {
    canvas: { background: canvasBg },
    node: {
      fill: '#7CA0AB',
      activeFill: '#1DE9AC',
      opacity: 1,
      selectedOpacity: 1,
      inactiveOpacity: 0.2,
      label: {
        color: `${text}`,
        stroke: `${canvasBg}`,
        activeColor: `${highlightText}`,
      },
      subLabel: {
        color: `${secondaryText}`,
        stroke: `${canvasBg}`,
        activeColor: `${highlightText}`,
      },
    },
    lasso: {
      border: '1px solid #55aaff',
      background: 'rgba(75, 160, 255, 0.1)',
    },
    ring: {
      fill: '#D8E6EA',
      activeFill: '#1DE9AC',
    },
    edge: {
      fill: '#D8E6EA',
      activeFill: '#1DE9AC',
      opacity: 1,
      selectedOpacity: 1,
      inactiveOpacity: 0.1,
      label: {
        stroke: `${canvasBg}`,
        color: `${text}`,
        activeColor: `${highlightText}`,
      },
    },
    arrow: {
      fill: '#D8E6EA',
      activeFill: '#1DE9AC',
    },
    cluster: {
      stroke: `${canvasBg}`,
      opacity: 1,
      selectedOpacity: 1,
      inactiveOpacity: 0.1,
      label: {
        stroke: `${canvasBg}`,
        color: '#2A6475',
      },
    },
  };
  return theme;
};

/* Design: Relationship Graph

  Description: Build a tree representing the related items for a particular sample (root node)

  Types of Relationships:
    - Parents
    - Children
    - Cousins

  Relationship Definitions:
    - Parent: A sample that will "drop" the sample when run through a tool
    - Child: A sample that is "dropped" by the sample when run through a tool
    - Cousins: A sample that is similar to the sample

  Tree Node Structure: Sample
    {
      sha256: <sha256>,
      type: parent/child/relation/root,
      Origin: [{type: <Unpacked/Transformed/MemoryDump>, parent: <sha256>}],
      Parents: [] or Null,
      Children: [] or Null,
      Cousins: [] or Null,
    }

  Functions:
    - buildSampleNode: build tree node recursively
*/

/** Class representing a related sample. */
class Sample {
  // eslint-disable-next-line require-jsdoc
  constructor(sha256, ancestors, descendants, isRoot) {
    this.sha256 = sha256;
    this.ancestors = ancestors;
    this.descendants = descendants;
    this.isRoot = isRoot;
    this.origin = [];
    this.tags = [];
    this.parents = [];
    this.children = [];
    this.cousins = [];
  }
}

// get tool results for a sample
const fetchResults = async (sha256) => {
  const resultsRes = await getResults(sha256, console.log, {});
  // return results if set
  if (resultsRes && 'results' in resultsRes) {
    return resultsRes.results;
  }
  return {};
};

// get submission info  for a sample
const fetchDetails = async (sha256) => {
  const detailsRes = await getFileDetails(sha256, console.log);
  // return just the submissions object if present
  if (detailsRes && 'submissions' in detailsRes && 'tags' in detailsRes) {
    return detailsRes;
  }
  // file details does not contain required keys
  return {};
};

// add a sample as a node to the nodes object
// we use an object to keep nodes unique
const addNode = (sha256, isRoot, info, nodes) => {
  if (sha256 in nodes) {
    return nodes;
  } else {
    nodes[sha256] = { id: sha256, isRoot: isRoot, info: info };
  }
};

// dump the nodes object as a list of nodes
// most graph libraries work with a list of nodes/links
const dumpFormattedNodes = (nodes) => {
  const d3FormattedNodes = [];
  Object.keys(nodes).map((sha256) => {
    d3FormattedNodes.push({
      id: sha256,
      isRoot: nodes[sha256]['isRoot'],
      label: `${sha256.slice(0, 6)}...`,
      info: nodes[sha256]['info'],
    });
  });
  return d3FormattedNodes;
};

// add a link to a list of network edges
const addLink = (source, target, links, type) => {
  if (source in links) {
    links[source].push({ sha256: target, type: type });
  } else {
    links[source] = [{ sha256: target, type: type }];
  }
  return links;
};

// dump a list of graph ready links objects
// this adds a label for the links
const dumpFormattedLinks = (links) => {
  const formattedLinks = [];
  Object.keys(links).map((source) => {
    links[source].map((target) => {
      formattedLinks.push({
        id: `${source}->${target.sha256}`,
        source: source,
        target: target.sha256,
        label: `${target.type}`,
      });
    });
  });
  return formattedLinks;
};

// recursively build the network data for a node
// uses DFS
const buildNetworkData = async (
  sha256,
  descendents,
  ancestors,
  isRoot,
  depth,
  nodes,
  links,
  bypassResults = {},
  bypassDetails = { tags: {}, submissions: [] },
) => {
  // grab presets if results
  const sample = new Sample(sha256, ancestors, descendents, isRoot);
  let results = bypassResults;
  let submissions = bypassDetails.submissions;
  let tags = bypassDetails.tags;

  if (!submissions.length) {
    const details = await fetchDetails(sha256);
    submissions = details.submissions;
    tags = details.tags;
  }

  // depth limit not yet reached
  if (!Object.keys(results).length) {
    results = await fetchResults(sha256);
  }

  // grab some tags to display next to the sample hash
  const includeTags = [
    'SymantecAV',
    'ClamAV',
    'YaraRuleHits',
    'FileType',
    'FileTypeExtension',
    'FileSize',
    'Dataset',
    'Family',
    'SampleType',
    'Imphash',
    'Att&ck',
  ];
  sample.tags = filterIncludedTags(tags, includeTags);

  // build a list of parents
  const parents = [];
  depth &&
    submissions &&
    Array.isArray(submissions) &&
    submissions.map((submission) => {
      if (submission && 'origin' in submission && submission['origin'] != 'None') {
        if ('Unpacked' in submission['origin']) {
          parents.push({
            sha256: submission['origin']['Unpacked']['parent'],
            tool: submission['origin']['Unpacked']['tool'],
            origin: 'Unpacked',
          });
        } else if ('Transformed' in submission['origin']) {
          parents.push({
            sha256: submission['origin']['Transformed']['parent'],
            tool: submission['origin']['Transformed']['tool'],
            origin: 'Transformed',
          });
        } else if ('MemoryDump' in submission['origin']) {
          parents.push({
            sha256: submission['origin']['MemoryDump']['parent'],
            tool: submission['origin']['MemoryDump']['tool'],
            origin: 'MemoryDump',
          });
        }
      }
    });

  // build a list of children and cousins
  const cousins = [];
  const children = [];
  depth &&
    results &&
    Object.keys(results).map((tool) => {
      if (tool && tool in results) {
        const result = results[tool][0];
        if ('children' in result && Object.keys(result['children']).length > 0) {
          Object.keys(result['children']).map((sha256) => {
            children.push({ tool: tool, sha256: sha256 });
          });
        }
      }
    });

  // ------ Generate relations for each relationship type -----
  // build children nodes
  const childrenPromises = [];
  children.map((child) => {
    if (![...descendents, ...ancestors, sha256].includes(child.sha256)) {
      childrenPromises.push(buildNetworkData(child.sha256, [...ancestors, sha256], [...descendents], false, depth - 1, nodes, links));
    } else if (isRoot) {
      childrenPromises.push(buildNetworkData(child.sha256, [], [], false, 0, nodes, links));
    }
  });
  sample.children.push(...(await Promise.all(childrenPromises)));

  // build related cousin nodes (aka similar samples)
  const cousinPromises = [];
  cousins.map((cousin) => {
    if (cousin.sha256 != sha256) {
      // do not recurse into related samples info, we can change this later
      cousinPromises.push(buildNetworkData(cousin.sha256, [], [], false, 0, nodes, links));
    }
  });
  sample.cousins.push(...(await Promise.all(cousinPromises)));

  // build children nodes
  const parentPromises = [];
  parents.map((parent) => {
    if (![...descendents, ...ancestors, sha256].includes(parent.sha256)) {
      parentPromises.push(buildNetworkData(parent.sha256, [...ancestors], [...descendents, sha256], false, depth - 1, nodes, links));
    } else if (isRoot) {
      // this is points back to itself, do not recurse further
      parentPromises.push(buildNetworkData(parent.sha256, [], [], false, 0, nodes, links));
    }
  });
  sample.parents.push(...(await Promise.all(parentPromises)));

  // the origins are the parents structure∂
  sample.origin = parents;

  // add current sha256 to graph nodes list
  addNode(sha256, isRoot, { tags: sample.tags, origin: sample.origin }, nodes);

  // Add parent relationships to D3 nodes
  parents.map((parent) => {
    addLink(parent.sha256, sha256, links, parent.tool);
  });

  // Add child relationships to D3 nodes
  children.map((child) => {
    addLink(sha256, child.sha256, links, child.tool);
  });

  // Add cousin related sample relationships to D3 nodes
  cousins.map((cousin) => {
    // bidirectional relationship since samples are related
    addLink(sha256, cousin.sha256, links, cousin.tool);
  });

  return sample;
};

const getUpdatedTheme = () => {
  // really hacky bs, don't be this bad

  let canvasBg = getComputedStyle(document.getElementById('thorium')).getPropertyValue('--thorium-panel-bg');
  if (canvasBg[0] != '#') {
    canvasBg = canvasBg
      .split('(')[1]
      .split(')')[0]
      .split(',')
      .map((part) => parseInt(part.trim()));
    canvasBg = `#${convert.rgb.hex(canvasBg)}`;
  }

  let text = getComputedStyle(document.getElementById('thorium')).getPropertyValue('--thorium-text');
  if (text[0] != '#') {
    text = text
      .split('(')[1]
      .split(')')[0]
      .split(',')
      .map((part) => parseInt(part.trim()));
    text = `#${convert.rgb.hex(text)}`;
  }

  let highlightText = getComputedStyle(document.getElementById('thorium')).getPropertyValue('--thorium-highlight-text');
  if (highlightText[0] != '#') {
    highlightText = highlightText
      .split('(')[1]
      .split(')')[0]
      .split(',')
      .map((part) => parseInt(part.trim()));
    highlightText = `#${convert.rgb.hex(highlightText)}`;
  }

  let secondaryText = getComputedStyle(document.getElementById('thorium')).getPropertyValue('--thorium-secondary-text');
  if (secondaryText[0] != '#') {
    secondaryText = secondaryText
      .split('(')[1]
      .split(')')[0]
      .split(',')
      .map((part) => parseInt(part.trim()));
    secondaryText = `#${convert.rgb.hex(secondaryText)}`;
  }
  return theme(canvasBg, text, highlightText, secondaryText);
};

// ---------------- Relationship tree component ----------------
const Related = ({ sha256, results, details }) => {
  const [theme, setTheme] = useState(getUpdatedTheme());
  const [maxDepth, setMaxDepth] = useState(1);
  const [nodes, setNodes] = useState([]);
  const [links, setLinks] = useState([]);
  // which node or edge is currently selected by the graph
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [showNodeLabels, setShowNodeLabels] = useState(true);
  const [showEdgeLabels, setShowEdgeLabels] = useState(true);
  const [graphLabelType, setGraphLabelType] = useState('all');
  const [showLegend, setShowLegend] = useState(false);

  // build and render the tree
  useEffect(() => {
    const buildTree = async () => {
      const nodes = {};
      const links = {};
      // build network data
      await buildNetworkData(sha256, [], [], true, maxDepth, nodes, links, results, details);
      // dump formatted nodes and links for rendering graph
      setNodes(dumpFormattedNodes(nodes));
      setLinks(dumpFormattedLinks(links));
    };
    buildTree();
    setTheme(getUpdatedTheme());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sha256, maxDepth, details, results]);

  // update the selected network graph element
  const updateSelectedElement = (element, type) => {
    switch (type) {
      case 'edge':
        setSelectedEdge(element);
        setSelectedNode(null);
        break;
      case 'node':
        setSelectedEdge(null);
        setSelectedNode(element);
        break;
      default:
        break;
    }
  };

  const Legend = () => {
    return (
      <div className="m-2">
        <p>What do graph nodes represent?</p>
        <p className="mx-3">
          The round circles displayed on the screen are the nodes of the graph. Nodes represent files that have a relationship with other
          files in the graph. The graph&apos;s relationships are built from tool results and the origin information of files that have been
          uploaded to Thorium. Incomplete or inaccurate origin information can result in an inaccurate relationship graph.
        </p>
        <p>What do the links between nodes represent?</p>
        <p className="mx-3">The link between two nodes represents a relationship that those two files share.</p>
        <p>What types of relationships are represented by the graph?</p>
        <p className="mx-3">
          Relationships between nodes can be direct or indirect. Ancestor or descendant files are direct relations of a sample. These
          relationships are created when an ancestor unpacks or is transformed (ie is patched) to produce a child sample. Indirect
          relationships come in the form of samples that share a similar characteristic. 
        </p>
        <hr />
      </div>
    );
  };

  const NodeInfo = ({ node }) => {
    const tags = node.info.tags;
    // const origin = node.info.origin;
    return (
      <div className="m-2">
        <Subtitle>SHA256</Subtitle>
        <Link to={`/file/${node.id}/`} target={'_blank'}>
          {node.id}
        </Link>
        {/* <hr/>
        <Subtitle className='mt-2'>Origin</Subtitle>*/}
        {/* Object.keys(origin).length == 0 &&
          <p className='ms-4'>N/A</p>*/}
        <hr />
        <Subtitle className="mt-2">tags</Subtitle>
        {Object.keys(tags).length == 0 && <p className="ms-4 text"> N/A</p>}
        <div className="d-flex justify-content-start wrap">
          {Object.keys(tags).map((tagKey) =>
            Object.keys(tags[tagKey]).map((tagValue) => (
              <TagBadge key={`${tagKey}_${tagValue}`} tag={tagKey} value={tagValue} condensed={true} action={'none'} />
            )),
          )}
        </div>
        <hr />
      </div>
    );
  };

  const updateLabels = (type) => {
    let edgeLabels = null;
    let nodeLabels = null;
    switch (type) {
      case 'edge':
        edgeLabels = !showEdgeLabels;
        nodeLabels = showNodeLabels;
        setShowEdgeLabels(edgeLabels);
        break;
      case 'node':
        nodeLabels = !showNodeLabels;
        edgeLabels = showEdgeLabels;
        setShowNodeLabels(nodeLabels);
        break;
    }
    if (nodeLabels) {
      if (edgeLabels) setGraphLabelType('all');
      else setGraphLabelType('nodes');
    } else {
      if (edgeLabels) setGraphLabelType('edges');
      else setGraphLabelType('none');
    }
  };

  return (
    <div id="related-tab">
      <div className="row">
        <div className="col d-flex justify-content-center">
          <Alert variant="info" className="d-flex justify-content-center mb-2 near-full-width">
            The relationship graph is a prototype feature. Please provide feedback or report issues to Thorium's &nbsp;
            <a href="mailto:wg-thorium-devs@sandia.gov">developers</a>.
          </Alert>
        </div>
      </div>
      <div className="row">
        <div className="col d-flex justify-content-center">
          <Card className="related-card">
            <GraphCanvas
              className="related-graph"
              labelType={graphLabelType}
              nodes={nodes}
              edges={links}
              edgeLabelPosition={'natural'}
              onNodeClick={(node) => updateSelectedElement(node.data, 'node')}
              onEdgeClick={(edge) => updateSelectedElement(edge.data, 'edge')}
              layoutOverrides={{
                linkDistance: 100,
                nodeStrength: -250,
                clusterPadding: 40,
                clusterStrength: 2,
              }}
              edgeInterpolation={'curved'}
              animated={true}
              theme={theme}
            />
            <div className="related-node-info">
              {showLegend && <Legend />}
              {selectedNode && <NodeInfo node={selectedNode} />}
              {selectedEdge && <pre className="text">{JSON.stringify(selectedEdge, null, 3)}</pre>}
              <div className="m-2">
                <Form className="ms-4 text">
                  <Form.Check
                    type="switch"
                    id="form-show-node-label"
                    label="Node Labels"
                    checked={showNodeLabels}
                    onChange={() => updateLabels('node')}
                  />
                  <Form.Check
                    type="switch"
                    id="form-show-edge-label"
                    label="Edge Labels"
                    checked={showEdgeLabels}
                    onChange={() => updateLabels('edge')}
                  />
                  <Form.Check
                    type="switch"
                    id="form-show-legend"
                    label="Show Help"
                    checked={showLegend}
                    onChange={() => setShowLegend(!showLegend)}
                  />
                  <Form.Label>Depth</Form.Label>
                  <Form.Select
                    id="form-select-depth-int"
                    value={maxDepth}
                    style={{ width: '75px' }}
                    onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                  >
                    {[1, 2, 3, 4, 5, 6, 7, 8].map((depth) => (
                      <option key={`${depth}`}>{depth}</option>
                    ))}
                  </Form.Select>
                </Form>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Related;
