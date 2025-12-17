export enum GraphLayout {
  'Fcose' = 'fcose',
  'CoseBilent' = 'cose-bilkent',
  'Elk' = 'elk',
  'Circle' = 'circle',
  'Concentric' = 'concentric',
  //'Cise' = 'cise',
}

// build layout props
export const getLayout = (layout: GraphLayout): cytoscape.LayoutOptions => {
  // https://github.com/iVis-at-Bilkent/cytoscape.js-fcose
  if (layout == GraphLayout.Fcose) {
    return {
      name: GraphLayout.Fcose,
      // 'draft', 'default' or 'proof'
      // - "draft" only applies spectral layout
      // - "default" improves the quality with incremental layout (fast cooling rate)
      // - "proof" improves the quality with incremental layout (slow cooling rate)
      //@ts-ignore
      quality: 'default',
      // Use random node positions at beginning of layout
      // if this is set to false, then quality option must be "proof"
      randomize: true,
      // Whether or not to animate the layout
      //@ts-ignore
      animate: true,
      // Duration of animation in ms, if enabled
      animationDuration: 500,
      // Easing of animation, if enabled
      animationEasing: undefined,
      // Fit the viewport to the repositioned nodes
      fit: true,
      // Padding around layout
      padding: 20,
      // Whether to include labels in node dimensions. Valid in "proof" quality
      uniformNodeDimensions: false,
      // Whether or not simple nodes (non-compound nodes) are of uniform dimensions
      nodeDimensionsIncludeLabels: false,
      // Whether to pack disconnected components - cytoscape-layout-utilities extension should be registered and initialized
      packComponents: false,

      /* spectral layout options */

      // False for random, true for greedy sampling
      samplingType: true,
      // Sample size to construct distance matrix
      sampleSize: 25,
      // Separation amount between nodes
      nodeSeparation: 200,
      // Power iteration tolerance
      piTol: 0.0000001,

      /* incremental layout options */

      // Node repulsion (non overlapping) multiplier
      // pushes nodes apparent and makes nodes on large graphs less likely to collide, but it will shrink nodes
      nodeRepulsion: 50000,
      //@ts-ignore
      idealEdgeLength: 200, // Ideal edge (non nested) length
      //@ts-ignore
      edgeElasticity: 0.75, // Divisor to compute edge forces
      // Nesting factor (multiplier) to compute ideal edge length for nested edges
      nestingFactor: 0.5,
      // Maximum number of iterations to perform - this is a suggested value and might be adjusted by the algorithm as required
      numIter: 2000,
      // For enabling tiling
      tile: true,
      // The comparison function to be used while sorting nodes during tiling operation.
      // Takes the ids of 2 nodes that will be compared as a parameter and the default tiling operation is performed when this option is not set.
      // It works similar to ``compareFunction`` parameter of ``Array.prototype.sort()``
      // If node1 is less then node2 by some ordering criterion ``tilingCompareBy(nodeId1, nodeId2)`` must return a negative value
      // If node1 is greater then node2 by some ordering criterion ``tilingCompareBy(nodeId1, nodeId2)`` must return a positive value
      // If node1 is equal to node2 by some ordering criterion ``tilingCompareBy(nodeId1, nodeId2)`` must return 0
      tilingCompareBy: undefined,
      // Represents the amount of the vertical space to put between the zero degree members during the tiling operation(can also be a function)
      tilingPaddingVertical: 10,
      // Represents the amount of the horizontal space to put between the zero degree members during the tiling operation(can also be a function)
      tilingPaddingHorizontal: 10,
      // Gravity force (constant)
      gravity: 0.25,
      // Gravity range (constant) for compounds
      gravityRangeCompound: 1.5,
      // Gravity force (constant) for compounds
      gravityCompound: 1.0,
      // Gravity range (constant)
      gravityRange: 3.8,
      // Initial cooling factor for incremental layout
      initialEnergyOnIncremental: 0.3,

      /* constraint options */

      // Fix desired nodes to predefined positions
      // [{nodeId: 'n1', position: {x: 100, y: 200}}, {...}]
      fixedNodeConstraint: undefined,
      // Align desired nodes in vertical/horizontal direction
      // {vertical: [['n1', 'n2'], [...]], horizontal: [['n2', 'n4'], [...]]}
      alignmentConstraint: undefined,
      // Place two nodes relatively in vertical/horizontal direction
      // [{top: 'n1', bottom: 'n2', gap: 100}, {left: 'n3', right: 'n4', gap: 75}, {...}]
      relativePlacementConstraint: undefined,
    };
  } else if (layout == GraphLayout.CoseBilent) {
    return {
      name: GraphLayout.CoseBilent,
      // 'draft', 'default' or 'proof"
      // - 'draft' fast cooling rate
      // - 'default' moderate cooling rate
      // - "proof" slow cooling rate
      //@ts-ignore
      quality: 'default',
      // Whether to include labels in node dimensions. Useful for avoiding label overlap
      //@ts-ignore
      nodeDimensionsIncludeLabels: false,
      // number of ticks per frame; higher is faster but more jerky
      refresh: 30,
      // Whether to fit the network view after when done
      fit: true,
      // Padding on fit
      padding: 20,
      // Whether to enable incremental mode
      //@ts-ignore
      randomize: true,
      // Node repulsion (non overlapping) multiplier
      //@ts-ignore
      nodeRepulsion: 50000,
      // Ideal (intra-graph) edge length
      //@ts-ignore
      idealEdgeLength: 200,
      // Divisor to compute edge forces
      //@ts-ignore
      edgeElasticity: 0.45,
      // Nesting factor (multiplier) to compute ideal edge length for inter-graph edges
      nestingFactor: 0.1,
      // Gravity force (constant)
      gravity: 0.25,
      // Maximum number of iterations to perform
      //@ts-ignore
      numIter: 2000,
      // Whether to tile disconnected nodes
      tile: true,
      // Type of layout animation. The option set is {'during', 'end', false}
      //@ts-ignore
      animate: 'end',
      // Duration for animate:end
      //@ts-ignore
      animationDuration: 500,
      // Amount of vertical space to put between degree zero nodes during tiling (can also be a function)
      tilingPaddingVertical: 10,
      // Amount of horizontal space to put between degree zero nodes during tiling (can also be a function)
      tilingPaddingHorizontal: 10,
      // Gravity range (constant) for compounds
      gravityRangeCompound: 1.5,
      // Gravity force (constant) for compounds
      gravityCompound: 1.0,
      // Gravity range (constant)
      gravityRange: 3.8,
      // Initial cooling factor for incremental layout
      initialEnergyOnIncremental: 0.5,
    };
  } else if (layout == GraphLayout.Circle) {
    return {
      name: GraphLayout.Circle, // we need to add clusters to structure before this one will work
      //@ts-ignore
      animate: false,
      fit: true,
      nodeDimensionsIncludeLabels: false,
    };
  } else if (layout == GraphLayout.Concentric) {
    return {
      name: GraphLayout.Concentric, // we need to add clusters to structure before this one will work
      //@ts-ignore
      animate: false,
      fit: true,
      nodeDimensionsIncludeLabels: false,
    };
  } else if (layout == GraphLayout.Elk) {
    return {
      name: GraphLayout.Elk, // we need to add clusters to structure before this one will work
      //@ts-ignore
      animate: true,
      fit: true,
      nodeDimensionsIncludeLabels: false,
      animationDuration: 500,
    };
  }
  return {
    name: layout,
    //@ts-ignore
    animate: true,
    animationDuration: 500,
    nodeDimensionsIncludeLabels: true,
    fit: true,
    padding: 20,
  };
};
