import React from 'react';
import styled from 'styled-components';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';

// project imports
import TagBadge from '../TagBadge';
import { filterIncludedTags, filterExcludedTags } from '../utilities';
import { Tags } from '@models/tags';
import { Entities } from '@models/entities';

const TagContainer = styled.div`
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
`;

interface CondensedEntityTagProps {
  tags: Tags; // tags to display in condensed non-editable view
  resource?: Entities;
}

const CondensedEntityTags: React.FC<CondensedEntityTagProps> = ({ tags, resource }) => {
  const excludeTags: string[] = [];
  const generalTags = filterExcludedTags(tags, excludeTags);
  const tlpTags = filterIncludedTags(tags, ['TLP']);
  const tagsCount = Object.keys(tags).length;
  return (
    <>
      {tagsCount == 0 && (
        <div className="px-3 py-2">
          <AlertBanner severity={Severity.Info} className="ms-4 me-4">
            No Tags Found
          </AlertBanner>
        </div>
      )}
      <TagContainer>
        {Object.keys(tlpTags).length > 0 &&
          Object.keys(tlpTags)
            .sort()
            .map((tagKey) =>
              Object.keys(tlpTags[tagKey])
                .sort()
                .map((tagValue) => (
                  <TagBadge resource={resource} key={'TLP_' + tagValue} tag={tagKey} value={tagValue} condensed={true} action={'link'} />
                )),
            )}
        {Object.keys(generalTags)
          .sort()
          .map((tagKey) =>
            Object.keys(generalTags[tagKey])
              .sort()
              .map((tagValue) => (
                <TagBadge resource={resource} key={'General_' + tagValue} tag={tagKey} value={tagValue} condensed={true} action={'link'} />
              )),
          )}
      </TagContainer>
    </>
  );
};

export default CondensedEntityTags;
