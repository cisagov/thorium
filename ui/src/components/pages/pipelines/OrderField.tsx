import React from 'react';
import styled from 'styled-components';

// project imports
import PipelineOrderFlow from '@components/shared/pipeline/PipelineOrderFlow';

const Note = styled.div`
  color: var(--thorium-secondary-text);
  font-size: 13px;
  margin: 4px 0;
`;

interface OrderFieldProps {
  /// The current pipeline order (stages of images, parallel stages are nested arrays)
  order: (string | string[])[];
  /// Called with the updated order whenever the user edits it
  onChange: (order: (string | string[])[]) => void;
  /// The group whose images can be added to the order
  group: string;
  /// The set of banned image names to flag in the diagram
  bannedImages?: Set<string>;
}

/// Editable pipeline order field. Renders the xydiagram (which now handles an empty
/// order in edit mode) so the first and subsequent images are all added on the diagram.
const OrderField: React.FC<OrderFieldProps> = ({ order, onChange, group, bannedImages }) => {
  // the diagram's add menu needs the group's image list, so a group must be chosen first
  if (!group) {
    return <Note>Select a group to configure the pipeline order.</Note>;
  }
  return (
    <>
      <Note>Double-click to add an image · right-click a node to insert/remove · drag a node to reorder.</Note>
      <Note>Drag the background to pan · scroll to zoom.</Note>
      <PipelineOrderFlow order={order} onOrderChange={onChange} bannedImages={bannedImages} group={group} />
    </>
  );
};

export default OrderField;
